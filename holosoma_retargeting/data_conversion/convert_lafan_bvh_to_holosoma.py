#!/usr/bin/env python3
"""Convert an original LAFAN BVH file to Holosoma ``lafan`` input.

The output is a ``(frames, 22, 3)`` float32 array in meters, Y-up, and in
``LAFAN_DEMO_JOINTS`` order.  Holosoma's LAFAN loader performs the final
Y-up-to-Z-up rotation.  Horizontal translation is removed using the first
frame Hips position.

This module contains the small subset of BVH parsing and quaternion forward
kinematics needed by LAFAN.  It intentionally has no dependency on the
external ``lafan1`` package used by the upstream extraction script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from holosoma_retargeting.config_types.data_type import LAFAN_DEMO_JOINTS
except ModuleNotFoundError as exc:
    if exc.name != "holosoma_retargeting":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from holosoma_retargeting.config_types.data_type import LAFAN_DEMO_JOINTS


CONVERSION_VERSION = 1
EXPECTED_FPS = 30.0
CM_TO_M = 0.01
JOINT_NAME_ALIASES = {
    "LeftToe": "LeftToeBase",
    "RightToe": "RightToeBase",
}
_ROTATION_CHANNEL_AXIS = {
    "Xrotation": 0,
    "Yrotation": 1,
    "Zrotation": 2,
}
_POSITION_CHANNEL_AXIS = {
    "Xposition": 0,
    "Yposition": 1,
    "Zposition": 2,
}


@dataclass(frozen=True)
class BvhMotion:
    joint_names: tuple[str, ...]
    parents: np.ndarray
    offsets_cm: np.ndarray
    local_positions_cm: np.ndarray
    local_quaternions_wxyz: np.ndarray
    frame_time: float


@dataclass(frozen=True)
class ConversionResult:
    input_bvh: Path
    output_npy: Path
    metadata_path: Path
    source_frames: int
    output_frames: int
    source_fps: float
    status: str
    diagnostics: dict[str, float | str]


def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product for arrays of scalar-first quaternions."""

    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )


def _quat_rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate vectors by scalar-first unit quaternions."""

    vector_part = quaternion[..., 1:]
    twice_cross = 2.0 * np.cross(vector_part, vector)
    return vector + quaternion[..., :1] * twice_cross + np.cross(vector_part, twice_cross)


def _axis_angle_quaternion(axis: int, angles_radians: np.ndarray) -> np.ndarray:
    quaternion = np.zeros((angles_radians.shape[0], 4), dtype=np.float64)
    half_angles = 0.5 * angles_radians
    quaternion[:, 0] = np.cos(half_angles)
    quaternion[:, axis + 1] = np.sin(half_angles)
    return quaternion


def _parse_hierarchy(lines: list[str], motion_line: int) -> tuple[
    tuple[str, ...], np.ndarray, np.ndarray, tuple[tuple[str, ...], ...]
]:
    joint_names: list[str] = []
    parents: list[int] = []
    offsets: list[np.ndarray] = []
    channels: list[tuple[str, ...]] = []
    active_joint = -1
    inside_end_site = False

    for raw_line in lines[:motion_line]:
        line = raw_line.strip()
        root_match = re.fullmatch(r"ROOT\s+(\S+)", line)
        joint_match = re.fullmatch(r"JOINT\s+(\S+)", line)
        if root_match or joint_match:
            name = (root_match or joint_match).group(1)
            joint_names.append(name)
            parents.append(active_joint)
            offsets.append(np.zeros(3, dtype=np.float64))
            channels.append(())
            active_joint = len(joint_names) - 1
            continue
        if line == "End Site":
            inside_end_site = True
            continue
        if line == "}":
            if inside_end_site:
                inside_end_site = False
            elif active_joint >= 0:
                active_joint = parents[active_joint]
            continue
        if inside_end_site or active_joint < 0:
            continue

        parts = line.split()
        if parts[:1] == ["OFFSET"]:
            if len(parts) != 4:
                raise ValueError(f"Invalid OFFSET line: {raw_line.rstrip()}")
            offsets[active_joint] = np.asarray(parts[1:4], dtype=np.float64)
        elif parts[:1] == ["CHANNELS"]:
            if len(parts) < 3 or int(parts[1]) != len(parts[2:]):
                raise ValueError(f"Invalid CHANNELS line: {raw_line.rstrip()}")
            channels[active_joint] = tuple(parts[2:])

    if not joint_names or parents[0] != -1:
        raise ValueError("BVH hierarchy has no valid ROOT")
    if any(not joint_channels for joint_channels in channels):
        missing = [name for name, value in zip(joint_names, channels) if not value]
        raise ValueError("BVH joints without channels: " + ", ".join(missing))
    return (
        tuple(joint_names),
        np.asarray(parents, dtype=np.int64),
        np.stack(offsets),
        tuple(channels),
    )


def read_lafan_bvh(input_bvh: Path | str) -> BvhMotion:
    """Read a LAFAN BVH file into local translations and rotations."""

    input_bvh = Path(input_bvh)
    if not input_bvh.is_file() or input_bvh.stat().st_size == 0:
        raise ValueError(f"BVH file is missing or empty: {input_bvh}")

    lines = input_bvh.read_text(encoding="utf-8").splitlines()
    try:
        motion_line = next(i for i, line in enumerate(lines) if line.strip() == "MOTION")
    except StopIteration as exc:
        raise ValueError(f"BVH file has no MOTION section: {input_bvh}") from exc

    joint_names, parents, offsets_cm, channels = _parse_hierarchy(lines, motion_line)
    if motion_line + 3 >= len(lines):
        raise ValueError(f"Incomplete MOTION header: {input_bvh}")

    frames_match = re.fullmatch(r"Frames:\s*(\d+)", lines[motion_line + 1].strip())
    frame_time_match = re.fullmatch(
        r"Frame Time:\s*([+\-0-9.eE]+)", lines[motion_line + 2].strip()
    )
    if frames_match is None or frame_time_match is None:
        raise ValueError(f"Invalid MOTION header: {input_bvh}")
    frame_count = int(frames_match.group(1))
    frame_time = float(frame_time_match.group(1))
    if frame_count <= 0 or not np.isfinite(frame_time) or frame_time <= 0:
        raise ValueError(f"Invalid frame count or frame time: {frame_count}, {frame_time}")

    channel_count = sum(len(joint_channels) for joint_channels in channels)
    motion_values = np.fromstring("\n".join(lines[motion_line + 3 :]), sep=" ", dtype=np.float64)
    expected_value_count = frame_count * channel_count
    if motion_values.size != expected_value_count:
        raise ValueError(
            f"Expected {expected_value_count} motion values, got {motion_values.size}: {input_bvh}"
        )
    motion_values = motion_values.reshape(frame_count, channel_count)

    local_positions_cm = np.broadcast_to(offsets_cm, (frame_count, *offsets_cm.shape)).copy()
    local_quaternions_wxyz = np.zeros((frame_count, len(joint_names), 4), dtype=np.float64)
    local_quaternions_wxyz[..., 0] = 1.0
    column = 0
    for joint_index, joint_channels in enumerate(channels):
        rotation_channels: list[tuple[int, np.ndarray]] = []
        for channel in joint_channels:
            values = motion_values[:, column]
            column += 1
            if channel in _POSITION_CHANNEL_AXIS:
                if joint_index != 0:
                    raise ValueError(
                        f"Only root translation channels are supported; found {channel} on "
                        f"{joint_names[joint_index]}"
                    )
                local_positions_cm[:, joint_index, _POSITION_CHANNEL_AXIS[channel]] = values
            elif channel in _ROTATION_CHANNEL_AXIS:
                rotation_channels.append((_ROTATION_CHANNEL_AXIS[channel], np.deg2rad(values)))
            else:
                raise ValueError(f"Unsupported BVH channel {channel!r}")

        rotation = local_quaternions_wxyz[:, joint_index]
        for axis, angles_radians in rotation_channels:
            rotation = _quat_multiply_wxyz(
                rotation,
                _axis_angle_quaternion(axis, angles_radians),
            )
        rotation /= np.linalg.norm(rotation, axis=1, keepdims=True)
        local_quaternions_wxyz[:, joint_index] = rotation

    return BvhMotion(
        joint_names=joint_names,
        parents=parents,
        offsets_cm=offsets_cm,
        local_positions_cm=local_positions_cm,
        local_quaternions_wxyz=local_quaternions_wxyz,
        frame_time=frame_time,
    )


def forward_kinematics(motion: BvhMotion) -> tuple[np.ndarray, np.ndarray]:
    """Return global scalar-first quaternions and positions in centimeters."""

    local_positions = motion.local_positions_cm
    local_quaternions = motion.local_quaternions_wxyz
    global_positions = np.empty_like(local_positions)
    global_quaternions = np.empty_like(local_quaternions)
    global_positions[:, 0] = local_positions[:, 0]
    global_quaternions[:, 0] = local_quaternions[:, 0]

    for joint_index in range(1, len(motion.joint_names)):
        parent = int(motion.parents[joint_index])
        if parent < 0 or parent >= joint_index:
            raise ValueError(
                f"Hierarchy is not parent-first at {motion.joint_names[joint_index]}: parent={parent}"
            )
        global_positions[:, joint_index] = (
            _quat_rotate_wxyz(global_quaternions[:, parent], local_positions[:, joint_index])
            + global_positions[:, parent]
        )
        global_quaternions[:, joint_index] = _quat_multiply_wxyz(
            global_quaternions[:, parent], local_quaternions[:, joint_index]
        )

    return global_quaternions, global_positions


def _normalized_joint_names(joint_names: Sequence[str]) -> list[str]:
    return [JOINT_NAME_ALIASES.get(name, name) for name in joint_names]


def _validate_and_reorder(
    motion: BvhMotion,
    global_positions_cm: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | str]]:
    source_names = _normalized_joint_names(motion.joint_names)
    missing = [name for name in LAFAN_DEMO_JOINTS if name not in source_names]
    if missing:
        raise ValueError("Missing required LAFAN joints: " + ", ".join(missing))
    positions_cm = global_positions_cm[
        :, [source_names.index(name) for name in LAFAN_DEMO_JOINTS], :
    ]
    if not np.all(np.isfinite(positions_cm)):
        raise ValueError("Global joint positions contain non-finite values")

    source_fps = 1.0 / motion.frame_time
    if not np.isclose(source_fps, EXPECTED_FPS, atol=0.05):
        raise ValueError(f"Expected a 30 FPS LAFAN BVH, got {source_fps:.6g} FPS")

    index = {name: i for i, name in enumerate(LAFAN_DEMO_JOINTS)}
    hips_to_head = positions_cm[:, index["Head"]] - positions_cm[:, index["Hips"]]
    hip_head_length = float(np.median(np.linalg.norm(hips_to_head, axis=1)))
    up_axis = "xyz"[int(np.argmax(np.median(np.abs(hips_to_head), axis=0)))]
    if not 40.0 <= hip_head_length <= 100.0:
        raise ValueError(
            f"Expected centimeter-scale skeleton; median Hips-Head={hip_head_length:.6g}"
        )
    if up_axis != "y":
        raise ValueError(f"Expected Y-up BVH coordinates; detected {up_axis.upper()}-up")

    toe_vectors = [
        positions_cm[:, index[f"{side}ToeBase"]] - positions_cm[:, index[f"{side}Foot"]]
        for side in ("Left", "Right")
    ]
    median_toe_length = float(
        np.median(np.linalg.norm(np.concatenate(toe_vectors, axis=0), axis=1))
    )
    if not 2.0 <= median_toe_length <= 35.0:
        raise ValueError(f"Invalid foot-to-toe length; median={median_toe_length:.6g}")

    forward = 0.5 * (toe_vectors[0] + toe_vectors[1])
    left = positions_cm[:, index["LeftUpLeg"]] - positions_cm[:, index["RightUpLeg"]]
    denominator = (
        np.linalg.norm(forward, axis=1)
        * np.linalg.norm(left, axis=1)
        * np.linalg.norm(hips_to_head, axis=1)
    )
    valid = denominator > 1e-8
    if not np.any(valid):
        raise ValueError("Cannot evaluate coordinate handedness")
    handedness = np.einsum(
        "ij,ij->i", np.cross(forward[valid], left[valid]), hips_to_head[valid]
    ) / denominator[valid]
    median_handedness = float(np.median(handedness))
    if median_handedness <= 0.2:
        raise ValueError(
            f"Unexpected forward/left/up coordinate handedness={median_handedness:.6g}"
        )

    diagnostics: dict[str, float | str] = {
        "source_fps": source_fps,
        "median_hips_head_cm": hip_head_length,
        "detected_up_axis": up_axis,
        "median_foot_toe_cm": median_toe_length,
        "median_forward_left_up_handedness": median_handedness,
    }
    return positions_cm, diagnostics


def convert_lafan_bvh_positions(
    input_bvh: Path | str,
) -> tuple[np.ndarray, dict[str, float | str], BvhMotion]:
    """Convert one BVH in memory to validated, centered Holosoma input."""

    motion = read_lafan_bvh(input_bvh)
    _, global_positions_cm = forward_kinematics(motion)
    positions_cm, diagnostics = _validate_and_reorder(motion, global_positions_cm)
    positions_m = positions_cm * CM_TO_M
    hips_index = LAFAN_DEMO_JOINTS.index("Hips")
    positions_m[:, :, 0] -= positions_m[0, hips_index, 0]
    positions_m[:, :, 2] -= positions_m[0, hips_index, 2]
    return positions_m.astype(np.float32), diagnostics, motion


def prepare_lafan_bvh_for_holosoma(
    input_bvh: Path | str,
    output_npy: Path | str,
    *,
    force: bool = False,
    write: bool = True,
) -> ConversionResult:
    """Validate and optionally write one converted BVH plus provenance metadata."""

    input_bvh = Path(input_bvh).resolve()
    output_npy = Path(output_npy).resolve()
    metadata_path = output_npy.with_suffix(".conversion.json")
    positions_m, diagnostics, motion = convert_lafan_bvh_positions(input_bvh)
    source_stat = input_bvh.stat()
    source_sha256 = hashlib.sha256(input_bvh.read_bytes()).hexdigest()
    metadata = {
        "conversion_version": CONVERSION_VERSION,
        "source_kind": "raw_lafan_bvh",
        "source_path": str(input_bvh),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "source_sha256": source_sha256,
        "source_frames": int(positions_m.shape[0]),
        "source_fps": diagnostics["source_fps"],
        "source_joint_names": list(motion.joint_names),
        "output_joint_names": list(LAFAN_DEMO_JOINTS),
        "source_coordinates": "centimeters_y_up",
        "generated_coordinates": "meters_y_up_first_frame_hips_xz_centered",
        "holosoma_y_up_to_z_up_matrix": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        "horizontal_centering": "subtract_first_frame_hips_xz",
        "diagnostics": diagnostics,
    }
    try:
        saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        saved_metadata = None
    current = output_npy.exists() and saved_metadata == metadata

    if write and (force or not current):
        output_npy.parent.mkdir(parents=True, exist_ok=True)
        temporary_npy = output_npy.with_name(output_npy.name + ".tmp")
        with temporary_npy.open("wb") as file_handle:
            np.save(file_handle, positions_m)
        temporary_npy.replace(output_npy)
        temporary_metadata = metadata_path.with_name(metadata_path.name + ".tmp")
        temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        temporary_metadata.replace(metadata_path)
        status = "written"
    elif current:
        status = "current"
    else:
        status = "stale_or_missing_not_written"

    return ConversionResult(
        input_bvh=input_bvh,
        output_npy=output_npy,
        metadata_path=metadata_path,
        source_frames=int(positions_m.shape[0]),
        output_frames=int(positions_m.shape[0]),
        source_fps=float(diagnostics["source_fps"]),
        status=status,
        diagnostics=diagnostics,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_bvh", type=Path, nargs="+", help="Original LAFAN BVH file(s)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for Holosoma .npy inputs")
    parser.add_argument("--force", action="store_true", help="Rewrite outputs even when metadata is current")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Parse and validate without writing output files",
    )
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()
    for input_bvh in args.input_bvh:
        output_npy = args.output_dir / f"{input_bvh.stem}.npy"
        result = prepare_lafan_bvh_for_holosoma(
            input_bvh,
            output_npy,
            force=args.force,
            write=not args.validate_only,
        )
        print(json.dumps(asdict(result), indent=2, default=str))


if __name__ == "__main__":
    main()
