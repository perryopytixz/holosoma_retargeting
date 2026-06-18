#!/usr/bin/env python3
"""Self-contained Plotly skeleton animation for retargeting variants."""

from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import tyro
import yourdfpy  # type: ignore[import-untyped]
from plotly.colors import qualitative

src_root = Path(__file__).resolve().parent.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))


SKELETON_LINKS = (
    "pelvis",
    "torso_link",
    "head_link",
    "left_hip_pitch_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "left_shoulder_pitch_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "left_rubber_hand_link",
    "right_shoulder_pitch_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
    "right_rubber_hand_link",
)

SKELETON_EDGES = (
    ("pelvis", "torso_link"),
    ("torso_link", "head_link"),
    ("pelvis", "left_hip_pitch_link"),
    ("left_hip_pitch_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_roll_link"),
    ("pelvis", "right_hip_pitch_link"),
    ("right_hip_pitch_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_roll_link"),
    ("torso_link", "left_shoulder_pitch_link"),
    ("left_shoulder_pitch_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_yaw_link"),
    ("left_wrist_yaw_link", "left_rubber_hand_link"),
    ("torso_link", "right_shoulder_pitch_link"),
    ("right_shoulder_pitch_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_yaw_link"),
    ("right_wrist_yaw_link", "right_rubber_hand_link"),
)

BOX_EDGE_IDS = (
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),
    (4, 5),
    (5, 7),
    (7, 6),
    (6, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


@dataclass(frozen=True)
class PlotlyVariantsConfig:
    """Configuration for self-contained Plotly variant animation export."""

    qpos_npzs: tuple[str, ...]
    """Retargeting result .npz files to compare."""

    export_html: str
    """Self-contained HTML output path."""

    labels: tuple[str, ...] = ()
    """Labels for each result. If empty, result parent names are used."""

    robot_urdf: str = "models/g1/g1_29dof.urdf"
    """Robot URDF path."""

    object_urdf: str | None = "models/largebox/largebox.urdf"
    """Object URDF path. Use None for robot-only results."""

    assume_object_in_qpos: bool = True
    """Whether qpos contains object pose in the final 7 entries."""

    layout: str = "grid"
    """Layout mode: grid or overlay."""

    spacing: float = 3.0
    """Distance between variants in grid layout."""

    columns: int = 4
    """Number of columns in grid layout."""

    fps: int = 30
    """Playback FPS."""

    frame_stride: int = 1
    """Export every Nth input frame."""

    include_plotlyjs: bool = True
    """Inline Plotly JavaScript so the HTML opens without network access."""

    title: str = "Retargeting Variant Animation"
    """Figure title."""


def _load_npz(path: str) -> tuple[np.ndarray, int]:
    data = np.load(path, allow_pickle=True)
    qpos = np.asarray(data["qpos"], dtype=float)
    fps = int(data["fps"]) if "fps" in data else 30
    return qpos, fps


def _infer_label(path: str) -> str:
    p = Path(path)
    parts = p.parts
    if "demo_results_total" in parts:
        idx = parts.index("demo_results_total")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return p.parent.name


def _layout_offsets(n: int, layout: str, spacing: float, columns: int) -> list[np.ndarray]:
    if layout == "overlay":
        return [np.zeros(3, dtype=float) for _ in range(n)]
    if layout != "grid":
        raise ValueError(f"Unsupported layout: {layout}")
    columns = max(1, int(columns))
    rows = int(math.ceil(n / columns))
    offsets: list[np.ndarray] = []
    x_center = 0.5 * (columns - 1)
    y_center = 0.5 * (rows - 1)
    for idx in range(n):
        row = idx // columns
        col = idx % columns
        offsets.append(np.array([(col - x_center) * spacing, (y_center - row) * spacing, 0.0], dtype=float))
    return offsets


def _quat_to_matrix(wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(wxyz, dtype=float)
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        return np.eye(3)
    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _available_skeleton(urdf: yourdfpy.URDF) -> tuple[list[str], list[tuple[int, int]]]:
    link_names = [name for name in SKELETON_LINKS if name in urdf.link_map]
    link_to_idx = {name: idx for idx, name in enumerate(link_names)}
    edges = [(link_to_idx[a], link_to_idx[b]) for a, b in SKELETON_EDGES if a in link_to_idx and b in link_to_idx]
    if not edges:
        raise ValueError("No configured skeleton links were found in the robot URDF.")
    return link_names, edges


def _joint_cfg(urdf: yourdfpy.URDF, q: np.ndarray, robot_dof: int) -> dict[str, float]:
    joints = q[7 : 7 + robot_dof]
    if joints.shape[0] != robot_dof:
        joints = joints[:robot_dof] if joints.shape[0] > robot_dof else np.pad(joints, (0, robot_dof - joints.shape[0]))
    return {name: float(value) for name, value in zip(urdf.actuated_joint_names, joints)}


def _robot_points(
    urdf: yourdfpy.URDF,
    q: np.ndarray,
    robot_dof: int,
    link_names: list[str],
    offset: np.ndarray,
) -> np.ndarray:
    urdf.update_cfg(_joint_cfg(urdf, q, robot_dof))
    local_points = np.array([urdf.get_transform(link_name)[:3, 3] for link_name in link_names], dtype=float)
    rotation = _quat_to_matrix(q[3:7])
    return local_points @ rotation.T + q[0:3] + offset


def _parse_mesh_scale(mesh_elem: ET.Element) -> np.ndarray:
    scale_text = mesh_elem.attrib.get("scale")
    if scale_text is None:
        return np.ones(3, dtype=float)
    values = [float(part) for part in scale_text.split()]
    if len(values) == 1:
        return np.array([values[0], values[0], values[0]], dtype=float)
    if len(values) != 3:
        raise ValueError(f"Invalid mesh scale: {scale_text}")
    return np.asarray(values, dtype=float)


def _object_box_corners(object_urdf: str | None) -> np.ndarray:
    default_extents = np.array([0.5, 0.5, 0.5], dtype=float)
    if object_urdf is None:
        extents = default_extents
        mins = -0.5 * extents
        maxs = 0.5 * extents
        return _corners_from_bounds(mins, maxs)

    urdf_path = Path(object_urdf)
    try:
        root = ET.parse(urdf_path).getroot()
        mesh_elem = root.find(".//mesh")
        if mesh_elem is None or "filename" not in mesh_elem.attrib:
            raise ValueError("No mesh geometry found.")
        mesh_path = Path(mesh_elem.attrib["filename"])
        if not mesh_path.is_absolute():
            mesh_path = urdf_path.parent / mesh_path
        scale = _parse_mesh_scale(mesh_elem)

        import trimesh  # type: ignore[import-untyped]

        mesh = trimesh.load(mesh_path, force="mesh")
        vertices = np.asarray(mesh.vertices, dtype=float) * scale
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
    except Exception:
        mins = -0.5 * default_extents
        maxs = 0.5 * default_extents
    return _corners_from_bounds(mins, maxs)


def _corners_from_bounds(mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [mins[0], mins[1], mins[2]],
            [maxs[0], mins[1], mins[2]],
            [mins[0], maxs[1], mins[2]],
            [maxs[0], maxs[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [maxs[0], mins[1], maxs[2]],
            [mins[0], maxs[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]],
        ],
        dtype=float,
    )


def _object_corners(q: np.ndarray, local_corners: np.ndarray, offset: np.ndarray) -> np.ndarray:
    rotation = _quat_to_matrix(q[-4:])
    return local_corners @ rotation.T + q[-7:-4] + offset


def _line_xyz(points: np.ndarray, edges: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> tuple[list, list, list]:
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for i, j in edges:
        xs.extend([float(points[i, 0]), float(points[j, 0]), None])
        ys.extend([float(points[i, 1]), float(points[j, 1]), None])
        zs.extend([float(points[i, 2]), float(points[j, 2]), None])
    return _rounded(xs), _rounded(ys), _rounded(zs)


def _rounded(values: list[float | None]) -> list[float | None]:
    return [None if value is None else round(float(value), 4) for value in values]


def _make_skeleton_trace(points: np.ndarray, edges: list[tuple[int, int]], label: str, color: str, showlegend: bool) -> go.Scatter3d:
    x, y, z = _line_xyz(points, edges)
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines+markers",
        name=label,
        legendgroup=label,
        showlegend=showlegend,
        line={"color": color, "width": 6},
        marker={"color": color, "size": 3},
    )


def _make_box_trace(corners: np.ndarray, label: str, color: str, showlegend: bool) -> go.Scatter3d:
    x, y, z = _line_xyz(corners, BOX_EDGE_IDS)
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        name=f"{label} object",
        legendgroup=label,
        showlegend=showlegend,
        line={"color": color, "width": 4},
        opacity=0.65,
    )


def _make_label_trace(label: str, offset: np.ndarray, color: str) -> go.Scatter3d:
    point = offset + np.array([0.0, -1.25, 2.1], dtype=float)
    return go.Scatter3d(
        x=[round(float(point[0]), 4)],
        y=[round(float(point[1]), 4)],
        z=[round(float(point[2]), 4)],
        mode="text",
        text=[label],
        textfont={"color": color, "size": 13},
        showlegend=False,
        hoverinfo="skip",
    )


def export_plotly_animation(cfg: PlotlyVariantsConfig) -> Path:
    if not cfg.qpos_npzs:
        raise ValueError("At least one --qpos-npzs path is required.")

    labels = list(cfg.labels) if cfg.labels else [_infer_label(path) for path in cfg.qpos_npzs]
    if len(labels) != len(cfg.qpos_npzs):
        raise ValueError("--labels must have the same length as --qpos-npzs.")

    loaded = [_load_npz(path) for path in cfg.qpos_npzs]
    qposes = [item[0] for item in loaded]
    n_frames = min(int(q.shape[0]) for q in qposes)
    if n_frames == 0:
        raise ValueError("At least one trajectory is empty.")
    frame_stride = max(1, int(cfg.frame_stride))
    frame_ids = list(range(0, n_frames, frame_stride))
    if frame_ids[-1] != n_frames - 1:
        frame_ids.append(n_frames - 1)
    qposes = [q[:n_frames] for q in qposes]

    urdf = yourdfpy.URDF.load(cfg.robot_urdf, load_meshes=False, build_scene_graph=True)
    robot_dof = len(urdf.actuated_joint_names)
    link_names, skeleton_edges = _available_skeleton(urdf)
    has_object = cfg.assume_object_in_qpos and cfg.object_urdf is not None and all(
        q.shape[1] >= 7 + robot_dof + 7 for q in qposes
    )
    object_corners_local = _object_box_corners(cfg.object_urdf) if has_object else None

    offsets = _layout_offsets(len(qposes), cfg.layout, cfg.spacing, cfg.columns)
    colors = qualitative.Dark24

    robot_frames: list[np.ndarray] = []
    object_frames: list[np.ndarray | None] = []
    all_points: list[np.ndarray] = []
    for qpos, offset in zip(qposes, offsets):
        variant_robot = np.stack(
            [_robot_points(urdf, qpos[frame_id], robot_dof, link_names, offset) for frame_id in frame_ids],
            axis=0,
        )
        robot_frames.append(variant_robot)
        all_points.append(variant_robot.reshape((-1, 3)))

        if has_object and object_corners_local is not None:
            variant_object = np.stack(
                [_object_corners(qpos[frame_id], object_corners_local, offset) for frame_id in frame_ids],
                axis=0,
            )
            object_frames.append(variant_object)
            all_points.append(variant_object.reshape((-1, 3)))
        else:
            object_frames.append(None)

    initial_data: list[go.Scatter3d] = []
    dynamic_trace_count = 0
    for idx, label in enumerate(labels):
        color = colors[idx % len(colors)]
        initial_data.append(_make_skeleton_trace(robot_frames[idx][0], skeleton_edges, label, color, True))
        dynamic_trace_count += 1
        if object_frames[idx] is not None:
            initial_data.append(_make_box_trace(object_frames[idx][0], label, color, False))
            dynamic_trace_count += 1

    for idx, (label, qpos, offset) in enumerate(zip(labels, qposes, offsets)):
        color = colors[idx % len(colors)]
        trajectory = qpos[:, 0:3] + offset
        initial_data.append(
            go.Scatter3d(
                x=np.round(trajectory[:, 0], 4),
                y=np.round(trajectory[:, 1], 4),
                z=np.round(trajectory[:, 2], 4),
                mode="lines",
                name=f"{label} base path",
                legendgroup=label,
                showlegend=False,
                line={"color": color, "width": 2, "dash": "dot"},
                opacity=0.35,
            )
        )
        initial_data.append(_make_label_trace(label, offset, color))

    frames: list[go.Frame] = []
    trace_ids = list(range(dynamic_trace_count))
    for local_frame_idx, frame_id in enumerate(frame_ids):
        frame_data: list[go.Scatter3d] = []
        for idx, label in enumerate(labels):
            color = colors[idx % len(colors)]
            frame_data.append(_make_skeleton_trace(robot_frames[idx][local_frame_idx], skeleton_edges, label, color, False))
            if object_frames[idx] is not None:
                frame_data.append(_make_box_trace(object_frames[idx][local_frame_idx], label, color, False))
        frames.append(go.Frame(data=frame_data, traces=trace_ids, name=str(frame_id)))

    bounds_points = np.vstack(all_points)
    mins = bounds_points.min(axis=0)
    maxs = bounds_points.max(axis=0)
    pad = max(0.5, 0.08 * float(np.max(maxs - mins)))
    mins -= pad
    maxs += pad

    frame_duration_ms = int(round(1000.0 * frame_stride / max(1, int(cfg.fps))))
    fig = go.Figure(data=initial_data, frames=frames)
    fig.update_layout(
        title=cfg.title,
        width=1280,
        height=820,
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
        scene={
            "xaxis": {"range": [float(mins[0]), float(maxs[0])], "title": "x"},
            "yaxis": {"range": [float(mins[1]), float(maxs[1])], "title": "y"},
            "zaxis": {"range": [float(mins[2]), float(maxs[2])], "title": "z"},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.45, "y": -1.75, "z": 1.05}},
        },
        legend={"itemsizing": "constant"},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": 0.03,
                "xanchor": "left",
                "yanchor": "bottom",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_duration_ms, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                                "mode": "immediate",
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.16,
                "y": 0.03,
                "len": 0.78,
                "currentvalue": {"prefix": "Frame "},
                "steps": [
                    {
                        "label": str(frame_id),
                        "method": "animate",
                        "args": [
                            [str(frame_id)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    }
                    for frame_id in frame_ids
                ],
            }
        ],
    )

    output_path = Path(cfg.export_html).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        output_path,
        include_plotlyjs=True if cfg.include_plotlyjs else "cdn",
        full_html=True,
        auto_play=False,
    )
    print(f"[plotly_variants_animation] Wrote self-contained HTML animation: {output_path}")
    print(
        "[plotly_variants_animation] "
        f"variants={len(labels)}, frames={len(frame_ids)}, robot_links={len(link_names)}, object={'yes' if has_object else 'no'}"
    )
    return output_path


def main(cfg: PlotlyVariantsConfig) -> None:
    export_plotly_animation(cfg)


if __name__ == "__main__":
    main(tyro.cli(PlotlyVariantsConfig))
