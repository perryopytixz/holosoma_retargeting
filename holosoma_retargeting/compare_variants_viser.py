#!/usr/bin/env python3
"""Synchronized Viser playback for multiple retargeting result variants."""

from __future__ import annotations

import math
import base64
import re
import socket
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
import zstandard  # type: ignore[import-untyped]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

src_root = Path(__file__).resolve().parent.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))


@dataclass(frozen=True)
class CompareVariantsConfig:
    """Configuration for synchronized variant comparison playback."""

    qpos_npzs: tuple[str, ...]
    """Retargeting result .npz files to compare."""

    labels: tuple[str, ...] = ()
    """Labels for each result. If empty, result parent names are used."""

    server_port: int = 8080
    """Port used by the interactive Viser viewer."""

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

    visual_fps_multiplier: int = 2
    """Visual interpolation multiplier."""

    loop: bool = True
    """Whether playback loops."""

    show_meshes: bool = True
    """Initial mesh visibility."""

    show_labels: bool = True
    """Initial label visibility."""

    grid_width: float = 14.0
    """Scene grid width."""

    grid_height: float = 10.0
    """Scene grid height."""

    variant_visibility: bool = True
    """Initial visibility for each variant."""

    export_html: str | None = None
    """Optional self-contained HTML animation path. If set, export and exit."""

    export_viewer_dir: str | None = None
    """Optional folder for an interactive offline viewer. If set, export and exit."""

    export_dark_mode: bool = False
    """Use a dark color scheme for exported HTML/viewer animations."""


@dataclass
class VariantPlayback:
    label: str
    qpos: np.ndarray
    fps: int
    offset: np.ndarray
    world_frame: viser.FrameHandle
    robot_base_frame: viser.FrameHandle
    object_base_frame: viser.FrameHandle | None
    label_handle: viser.LabelHandle
    robot: ViserUrdf
    obj: ViserUrdf | None
    visible: bool = True
    prev_robot_quat: np.ndarray | None = field(default=None, repr=False)
    prev_object_quat: np.ndarray | None = field(default=None, repr=False)


@dataclass
class ComparePlaybackContext:
    server: viser.ViserServer
    variants: list[VariantPlayback]
    n_frames: int
    robot_dof: int
    has_object: bool

    def reset_quat_continuity(self) -> None:
        for variant in self.variants:
            variant.prev_robot_quat = None
            variant.prev_object_quat = None

    def apply_discrete_frame(self, frame_idx: int) -> None:
        frame_idx = int(np.clip(frame_idx, 0, self.n_frames - 1))
        for variant in self.variants:
            _apply_frame(variant, variant.qpos[frame_idx], self.robot_dof, self.has_object)

    def apply_fractional_frame(self, frame_value: float) -> int:
        frame_value = float(np.clip(frame_value, 0.0, float(self.n_frames - 1)))
        k0 = int(np.floor(frame_value))
        k1 = min(k0 + 1, self.n_frames - 1)
        u = float(frame_value - k0)
        for variant in self.variants:
            q_interp = _interp_frame(variant.qpos, k0, k1, u, self.robot_dof, self.has_object)
            _apply_frame(variant, q_interp, self.robot_dof, self.has_object)
        return k0


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


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    norm = float(np.linalg.norm(q))
    return q if norm == 0.0 else q / norm


def _quat_continuous(prev_q: np.ndarray | None, curr_q: np.ndarray) -> np.ndarray:
    q = _quat_normalize(curr_q)
    if prev_q is None:
        return q
    return -q if float(np.dot(prev_q, q)) < 0.0 else q


def _slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return _quat_normalize(q0 + u * (q1 - q0))
    theta = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta = float(np.sin(theta))
    return (np.sin((1.0 - u) * theta) * q0 + np.sin(u * theta) * q1) / sin_theta


def _interp_frame(qpos: np.ndarray, i0: int, i1: int, u: float, robot_dof: int, has_object: bool) -> np.ndarray:
    q0 = qpos[i0]
    q1 = qpos[i1]
    out = q0.copy()
    out[0:3] = (1.0 - u) * q0[0:3] + u * q1[0:3]
    out[3:7] = _slerp(q0[3:7], q1[3:7], u)
    out[7 : 7 + robot_dof] = (1.0 - u) * q0[7 : 7 + robot_dof] + u * q1[7 : 7 + robot_dof]
    if has_object:
        out[-7:-4] = (1.0 - u) * q0[-7:-4] + u * q1[-7:-4]
        out[-4:] = _slerp(q0[-4:], q1[-4:], u)
    return out


def _set_variant_visible(variant: VariantPlayback, visible: bool, show_meshes: bool, show_label: bool) -> None:
    variant.visible = visible
    variant.world_frame.visible = visible
    variant.robot_base_frame.visible = visible
    if variant.object_base_frame is not None:
        variant.object_base_frame.visible = visible
    variant.robot.show_visual = visible and show_meshes
    if variant.obj is not None:
        variant.obj.show_visual = visible and show_meshes
    variant.label_handle.visible = visible and show_label


def _apply_frame(variant: VariantPlayback, q: np.ndarray, robot_dof: int, has_object: bool) -> None:
    joints = q[7 : 7 + robot_dof]
    if joints.shape[0] != robot_dof:
        joints = joints[:robot_dof] if joints.shape[0] > robot_dof else np.pad(joints, (0, robot_dof - joints.shape[0]))
    variant.robot.update_cfg(joints)

    variant.robot_base_frame.position = q[0:3]
    r_q = _quat_continuous(variant.prev_robot_quat, q[3:7])
    variant.prev_robot_quat = r_q
    variant.robot_base_frame.wxyz = r_q

    if has_object and variant.object_base_frame is not None:
        variant.object_base_frame.position = q[-7:-4]
        o_q = _quat_continuous(variant.prev_object_quat, q[-4:])
        variant.prev_object_quat = o_q
        variant.object_base_frame.wxyz = o_q


def _configure_initial_camera(
    server: viser.ViserServer,
    qposes: list[np.ndarray],
    offsets: list[np.ndarray],
    has_object: bool,
) -> None:
    scene_points: list[np.ndarray] = []
    label_offset = np.array([0.0, -1.25, 2.0], dtype=float)
    for qpos, offset in zip(qposes, offsets):
        scene_points.append(qpos[:, 0:3] + offset)
        scene_points.append(offset[None, :] + label_offset)
        if has_object:
            scene_points.append(qpos[:, -7:-4] + offset)

    points = np.vstack(scene_points)
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.size == 0:
        server.initial_camera.position = (6.0, -7.0, 4.5)
        server.initial_camera.look_at = (0.0, 0.0, 0.8)
        server.initial_camera.up = (0.0, 0.0, 1.0)
        return

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    mins[2] = min(mins[2], 0.0)
    maxs[2] = max(maxs[2], 2.3)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    distance = max(4.0, 1.35 * span)
    position = center + np.array([0.85 * distance, -1.0 * distance, 0.62 * distance], dtype=float)

    server.initial_camera.position = tuple(float(v) for v in position)
    server.initial_camera.look_at = tuple(float(v) for v in center)
    server.initial_camera.up = (0.0, 0.0, 1.0)


def _make_compat_html_text(html_text: str) -> str:
    script_match = re.search(r"<script\s+([^>]*)>.*?</script>", html_text, flags=re.DOTALL)
    if script_match is None:
        return html_text

    attrs = dict(re.findall(r'([\w-]+)="([^"]*)"', script_match.group(1)))
    required_attrs = ("data-s", "data-ss", "data-c", "data-cs")
    if not all(key in attrs for key in required_attrs):
        return html_text

    decompressor = zstandard.ZstdDecompressor()
    css = decompressor.decompress(base64.b64decode(attrs["data-s"]), max_output_size=int(attrs["data-ss"])).decode(
        "utf-8"
    )
    js = decompressor.decompress(base64.b64decode(attrs["data-c"]), max_output_size=int(attrs["data-cs"])).decode(
        "utf-8"
    )

    css = css.replace("</style", "<\\/style")
    js = js.replace("</script", "<\\/script")
    replacement = f"<style>{css}</style><script type=\"module\">{js}</script>"
    return html_text[: script_match.start()] + replacement + html_text[script_match.end() :]


def _make_compat_html(serializer: object, *, dark_mode: bool) -> str:
    html_text = serializer.as_html(dark_mode=dark_mode)  # type: ignore[attr-defined]
    return _make_compat_html_text(html_text)


def _record_scene_bytes(cfg: CompareVariantsConfig) -> tuple[bytes, int, float]:
    context = make_compare_scene(cfg, headless=True)
    try:
        context.reset_quat_continuity()
        context.apply_discrete_frame(0)
        serializer = context.server.get_scene_serializer()

        fps = max(1, int(cfg.fps))
        interp_mult = max(1, int(cfg.visual_fps_multiplier))
        dt = 1.0 / float(fps * interp_mult)
        n_steps = max(0, (context.n_frames - 1) * interp_mult)
        for step in range(1, n_steps + 1):
            serializer.insert_sleep(dt)
            context.apply_fractional_frame(step / interp_mult)

        return serializer.serialize(), n_steps, n_steps * dt
    finally:
        context.server.stop()


def _viser_client_html_path() -> Path:
    return Path(viser.__file__).resolve().parent / "client" / "build" / "index.html"


def _viewer_dir_index_html(*, playback_file: str, dark_mode: bool) -> str:
    client_html = _viser_client_html_path().read_text(encoding="utf-8")
    params = f"playbackPath={playback_file}"
    if dark_mode:
        params += "&darkMode"
    redirect_script = (
        "<script>"
        "(function(){"
        f"var params='{params}';"
        "if(!window.location.search){"
        "window.location.replace(window.location.pathname+'?'+params+window.location.hash);"
        "}"
        "})();"
        "</script>"
    )
    head_end = client_html.index("</head>")
    return _make_compat_html_text(client_html[:head_end] + redirect_script + client_html[head_end:])


def _serialized_scene_html(scene_bytes: bytes, *, dark_mode: bool) -> str:
    client_html = _viser_client_html_path().read_text(encoding="utf-8")
    scene_b64 = base64.b64encode(scene_bytes).decode("ascii")
    dark_mode_str = "true" if dark_mode else "false"
    inject_script = (
        f"<script>"
        f'window.__VISER_EMBED_DATA__="{scene_b64}";'
        f"window.__VISER_EMBED_CONFIG__={{darkMode:{dark_mode_str}}};"
        f"</script>"
    )
    head_end = client_html.index("</head>")
    return _make_compat_html_text(client_html[:head_end] + inject_script + client_html[head_end:])


def _viewer_dir_server_script() -> str:
    return """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import webbrowser
from pathlib import Path


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def bind_server(host: str, port: int, handler: type[http.server.SimpleHTTPRequestHandler]):
    if port == 0:
        return ReusableTCPServer((host, port), handler)

    last_error = None
    for candidate in range(port, port + 20):
        try:
            return ReusableTCPServer((host, candidate), handler)
        except OSError as exc:
            last_error = exc
            if exc.errno not in {98, 48, 10048}:
                break

    raise OSError(f"Could not bind {host}:{port}-{port + 19}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve this Viser offline viewer folder.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    handler = functools.partial(QuietHandler, directory=str(root))
    with bind_server(args.host, args.port, handler) as httpd:
        host, port = httpd.server_address
        url = f"http://{host}:{port}/index.html"
        print(url)
        if not args.no_open:
            webbrowser.open(url)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
"""


def make_compare_scene(cfg: CompareVariantsConfig, *, headless: bool = False) -> ComparePlaybackContext:
    if not cfg.qpos_npzs:
        raise ValueError("At least one --qpos-npzs path is required.")

    labels = list(cfg.labels) if cfg.labels else [_infer_label(path) for path in cfg.qpos_npzs]
    if len(labels) != len(cfg.qpos_npzs):
        raise ValueError("--labels must have the same length as --qpos-npzs.")

    loaded = [_load_npz(path) for path in cfg.qpos_npzs]
    qposes = [item[0] for item in loaded]
    fps_values = [item[1] for item in loaded]
    n_frames = min(int(q.shape[0]) for q in qposes)
    if n_frames == 0:
        raise ValueError("At least one trajectory is empty.")
    qposes = [q[:n_frames] for q in qposes]

    server = viser.ViserServer(port=_find_free_port() if headless else cfg.server_port, verbose=not headless)
    server.scene.add_grid("/grid", width=cfg.grid_width, height=cfg.grid_height, position=(0.0, 0.0, 0.0))

    robot_urdf = yourdfpy.URDF.load(cfg.robot_urdf, load_meshes=True, build_scene_graph=True)
    object_urdf = (
        yourdfpy.URDF.load(cfg.object_urdf, load_meshes=True, build_scene_graph=True) if cfg.object_urdf else None
    )

    probe_robot = ViserUrdf(server, urdf_or_path=robot_urdf, root_node_name="/_probe_robot")
    robot_dof = len(probe_robot.get_actuated_joint_limits())
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Attempted to remove already removed node: /_probe_robot.*")
        probe_robot.remove()

    has_object = cfg.assume_object_in_qpos and cfg.object_urdf is not None and all(
        q.shape[1] >= 7 + robot_dof + 7 for q in qposes
    )

    offsets = _layout_offsets(len(qposes), cfg.layout, cfg.spacing, cfg.columns)
    variants: list[VariantPlayback] = []
    for idx, (path, label, qpos, fps, offset) in enumerate(zip(cfg.qpos_npzs, labels, qposes, fps_values, offsets)):
        root = f"/variants/{idx:02d}_{label}"
        world_frame = server.scene.add_frame(root, show_axes=False, position=offset)
        robot_base_frame = server.scene.add_frame(f"{root}/robot", show_axes=False)
        robot = ViserUrdf(server, urdf_or_path=robot_urdf, root_node_name=f"{root}/robot")

        object_base_frame = None
        obj = None
        if cfg.object_urdf:
            object_base_frame = server.scene.add_frame(f"{root}/object", show_axes=False)
            obj = ViserUrdf(server, urdf_or_path=object_urdf, root_node_name=f"{root}/object")

        label_handle = server.scene.add_label(
            f"{root}/label",
            label,
            position=np.array([0.0, -1.25, 2.0]),
            font_size_mode="screen",
            font_screen_scale=0.9,
            anchor="center-center",
        )
        variant = VariantPlayback(
            label=label,
            qpos=qpos,
            fps=fps,
            offset=offset,
            world_frame=world_frame,
            robot_base_frame=robot_base_frame,
            object_base_frame=object_base_frame,
            label_handle=label_handle,
            robot=robot,
            obj=obj,
            visible=cfg.variant_visibility,
        )
        _set_variant_visible(variant, cfg.variant_visibility, cfg.show_meshes, cfg.show_labels)
        variants.append(variant)

    context = ComparePlaybackContext(
        server=server,
        variants=variants,
        n_frames=n_frames,
        robot_dof=robot_dof,
        has_object=has_object,
    )
    _configure_initial_camera(server, qposes, offsets, has_object)
    context.apply_discrete_frame(0)
    return context


def make_compare_player(cfg: CompareVariantsConfig) -> viser.ViserServer:
    context = make_compare_scene(cfg)
    server = context.server
    variants = context.variants
    n_frames = context.n_frames

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider("Frame", min=0, max=n_frames - 1, step=1, initial_value=0)
        play_btn = server.gui.add_button("Play / Pause")
        fps_in = server.gui.add_number("FPS", initial_value=cfg.fps, min=1, max=240, step=1)
        loop_cb = server.gui.add_checkbox("Loop", initial_value=cfg.loop)
    with server.gui.add_folder("Display"):
        show_meshes_cb = server.gui.add_checkbox("Show meshes", initial_value=cfg.show_meshes)
        show_labels_cb = server.gui.add_checkbox("Show labels", initial_value=cfg.show_labels)
    with server.gui.add_folder("Smoothing"):
        interp_mult_in = server.gui.add_number(
            "Visual FPS multiplier", initial_value=cfg.visual_fps_multiplier, min=1, max=8, step=1
        )
    with server.gui.add_folder("Variants"):
        variant_toggles = [
            server.gui.add_checkbox(variant.label, initial_value=cfg.variant_visibility) for variant in variants
        ]

    playing = {"flag": False}
    fractional_frame = {"value": 0.0}
    tick = {"next": time.perf_counter()}
    programmatic_slider_update = {"flag": False}

    @play_btn.on_click
    def _(_evt) -> None:
        playing["flag"] = not playing["flag"]
        tick["next"] = time.perf_counter()
        fractional_frame["value"] = float(frame_slider.value)
        context.reset_quat_continuity()

    @frame_slider.on_update
    def _(_evt) -> None:
        if programmatic_slider_update["flag"]:
            return
        playing["flag"] = False
        frame_idx = int(frame_slider.value)
        fractional_frame["value"] = float(frame_idx)
        context.reset_quat_continuity()
        context.apply_discrete_frame(frame_idx)

    @fps_in.on_update
    def _(_evt) -> None:
        tick["next"] = time.perf_counter()

    @interp_mult_in.on_update
    def _(_evt) -> None:
        tick["next"] = time.perf_counter()

    @show_meshes_cb.on_update
    def _(_evt) -> None:
        for variant, toggle in zip(variants, variant_toggles):
            _set_variant_visible(variant, bool(toggle.value), bool(show_meshes_cb.value), bool(show_labels_cb.value))

    @show_labels_cb.on_update
    def _(_evt) -> None:
        for variant, toggle in zip(variants, variant_toggles):
            _set_variant_visible(variant, bool(toggle.value), bool(show_meshes_cb.value), bool(show_labels_cb.value))

    for variant, toggle in zip(variants, variant_toggles):

        @toggle.on_update
        def _(_evt, variant=variant, toggle=toggle) -> None:
            _set_variant_visible(variant, bool(toggle.value), bool(show_meshes_cb.value), bool(show_labels_cb.value))

    def player_loop() -> None:
        if n_frames <= 1:
            return
        while True:
            if not playing["flag"]:
                time.sleep(0.02)
                continue
            now = time.perf_counter()
            fps_val = max(1, int(fps_in.value))
            interp_mult = max(1, int(interp_mult_in.value))
            dt = 1.0 / (fps_val * interp_mult)
            if now < tick["next"]:
                time.sleep(min(0.002, max(0.0, tick["next"] - now)))
                continue

            next_frame = fractional_frame["value"] + 1.0 / interp_mult
            if bool(loop_cb.value):
                next_frame = next_frame % max(1, n_frames)
            else:
                next_frame = min(next_frame, float(n_frames - 1))
                if next_frame >= n_frames - 1:
                    playing["flag"] = False
            fractional_frame["value"] = next_frame
            display_frame = context.apply_fractional_frame(next_frame)
            programmatic_slider_update["flag"] = True
            frame_slider.value = display_frame
            programmatic_slider_update["flag"] = False
            tick["next"] = now + dt

    threading.Thread(target=player_loop, daemon=True).start()

    print(f"[compare_variants_viser] Loaded {len(variants)} variants, {n_frames} synchronized frames.")
    print(f"[compare_variants_viser] robot_dof={context.robot_dof}, object={'yes' if context.has_object else 'no'}")
    print("[compare_variants_viser] Open the viewer URL printed above. Close the process with Ctrl+C.")
    return server


def export_compare_html(cfg: CompareVariantsConfig, html_path: str | Path) -> Path:
    output_path = Path(html_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scene_bytes, n_steps, duration = _record_scene_bytes(cfg)
    output_path.write_text(_serialized_scene_html(scene_bytes, dark_mode=cfg.export_dark_mode), encoding="utf-8")

    print(f"[compare_variants_viser] Wrote self-contained HTML animation: {output_path}")
    print(f"[compare_variants_viser] Recorded {n_steps + 1} visual states over {duration:.3f} s.")
    return output_path


def export_compare_viewer_dir(cfg: CompareVariantsConfig, viewer_dir: str | Path) -> Path:
    output_dir = Path(viewer_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_bytes, n_steps, duration = _record_scene_bytes(cfg)
    scene_path = output_dir / "scene.viser"
    scene_path.write_bytes(scene_bytes)

    index_path = output_dir / "index.html"
    index_path.write_text(
        _viewer_dir_index_html(playback_file=scene_path.name, dark_mode=cfg.export_dark_mode),
        encoding="utf-8",
    )

    server_path = output_dir / "serve_viewer.py"
    server_path.write_text(_viewer_dir_server_script(), encoding="utf-8")
    try:
        server_path.chmod(0o755)
    except OSError:
        pass

    readme_path = output_dir / "README.txt"
    readme_path.write_text(
        "\n".join(
            [
                "Interactive Viser offline viewer",
                "",
                "Recommended:",
                "  python3 serve_viewer.py",
                "",
                "If your system only has python on PATH:",
                "  python serve_viewer.py",
                "",
                "Then open the printed local URL in a browser.",
                "",
                "Files:",
                "  index.html    Viser web client",
                "  scene.viser   recorded mesh animation data",
                "  serve_viewer.py local static file server",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[compare_variants_viser] Wrote interactive viewer folder: {output_dir}")
    print(f"[compare_variants_viser] Open with: python3 {server_path}")
    print(f"[compare_variants_viser] Recorded {n_steps + 1} visual states over {duration:.3f} s.")
    print(f"[compare_variants_viser] scene.viser size: {scene_path.stat().st_size / (1024 * 1024):.1f} MiB")
    return output_dir


def main(cfg: CompareVariantsConfig) -> None:
    if cfg.export_viewer_dir is not None:
        export_compare_viewer_dir(cfg, cfg.export_viewer_dir)
        return

    if cfg.export_html is not None:
        export_compare_html(cfg, cfg.export_html)
        return

    make_compare_player(cfg)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(CompareVariantsConfig))
