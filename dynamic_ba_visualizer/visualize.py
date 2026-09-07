#!/usr/bin/env python3
"""Normalize model outputs and create Rerun/MP4/PNG diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scene_export import export_scenes, load_scenes, log_camera, new_recording


COLORS_BGR = {"batrack": (80, 180, 255), "droid_w": (80, 220, 80), "wildpose": (255, 150, 60)}
COLORS_RGB = {k: tuple(reversed(v)) for k, v in COLORS_BGR.items()}
LABELS = {"batrack": "BA-Track", "droid_w": "DROID-W", "wildpose": "WildPose"}


def quat_matrix(q):
    x, y, z, w = q
    n = x*x + y*y + z*z + w*w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w)],
        [s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w)],
        [s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)],
    ])


def load_tum(path: Path) -> np.ndarray:
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] < 8:
        raise ValueError(f"expected TUM trajectory with 8 columns: {path}")
    poses = np.repeat(np.eye(4)[None], len(data), axis=0)
    poses[:, :3, 3] = data[:, 1:4]
    for i, q in enumerate(data[:, 4:8]):
        poses[i, :3, :3] = quat_matrix(q)
    return poses


def normalize_poses(poses: np.ndarray):
    rel = np.linalg.inv(poses[0]) @ poses
    xyz = rel[:, :3, 3]
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    nonzero = steps[steps > 1e-6]
    scale = 1.0 / np.median(nonzero) if len(nonzero) else 1.0
    # Median-step normalization preserves drift/shape while removing monocular gauge scale.
    rel[:, :3, 3] *= scale
    return rel, scale


def model_paths(output: Path):
    return {
        "batrack": output / "models/batrack/sequence/batrack_traj.txt",
        "droid_w": output / "models/droid_w/sequence/traj/est_poses_full.txt",
        "wildpose": output / "models/wildpose/sequence/traj/est_poses_full.txt",
    }


def load_models(output: Path, requested: list[str]):
    models = {}
    for name, path in model_paths(output).items():
        if name not in requested or not path.exists():
            continue
        try:
            raw = load_tum(path)
            norm, scale = normalize_poses(raw)
            models[name] = {"path": str(path), "raw": raw, "poses": norm, "scale": scale}
        except Exception as exc:
            print(f"warning: cannot load {name}: {exc}")
    if not models:
        raise RuntimeError("no successful model trajectories were found")
    return models


def fit_points(points, rect, padding=14):
    x0, y0, x1, y1 = rect
    pts = np.asarray(points, np.float64)
    if not len(pts):
        return np.zeros((0, 2), np.int32)
    lo, hi = pts.min(0), pts.max(0)
    span = np.maximum(hi - lo, 1e-6)
    sx = (x1 - x0 - 2*padding) / span[0]
    sy = (y1 - y0 - 2*padding) / span[1]
    s = min(sx, sy)
    center = (lo + hi) / 2
    target = np.array([(x0+x1)/2, (y0+y1)/2])
    out = (pts - center) * np.array([s, -s]) + target
    return np.round(out).astype(np.int32)


def letterbox(image, width, height):
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1]*scale)), max(1, round(image.shape[0]*scale))))
    canvas = np.full((height, width, 3), 20, np.uint8)
    x = (width-resized.shape[1])//2; y = (height-resized.shape[0])//2
    canvas[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
    return canvas


def draw_panel(frame, name, poses, idx, width, height):
    panel = letterbox(frame, width, height)
    overlay_h = max(150, int(height * 0.38))
    y0 = height - overlay_h
    shade = panel.copy(); shade[y0:] = (10, 10, 10)
    panel = cv2.addWeighted(panel, 0.78, shade, 0.22, 0)
    cv2.rectangle(panel, (0, 0), (width, 42), (12, 12, 12), -1)
    cv2.putText(panel, LABELS[name], (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                COLORS_BGR[name], 2, cv2.LINE_AA)
    upto = min(idx, len(poses)-1)
    xyz = poses[:, :3, 3]
    pts = fit_points(xyz[:, [0, 2]], (8, y0+8, width-8, height-8), 12)
    if len(pts) > 1:
        cv2.polylines(panel, [pts], False, (85, 85, 85), 2, cv2.LINE_AA)
        cv2.polylines(panel, [pts[:upto+1]], False, COLORS_BGR[name], 3, cv2.LINE_AA)
        cv2.circle(panel, tuple(pts[upto]), 6, (255, 255, 255), -1, cv2.LINE_AA)
    if upto:
        step = np.linalg.norm(xyz[upto] - xyz[upto-1])
        cv2.putText(panel, f"frame {idx:06d}  normalized step {step:.3f}", (14, y0+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235,235,235), 1, cv2.LINE_AA)
    cv2.putText(panel, "XZ trajectory (monocular scale normalized)", (14, height-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210,210,210), 1, cv2.LINE_AA)
    return panel


def draw_source(frame, idx, width, height):
    panel = letterbox(frame, width, height)
    cv2.rectangle(panel, (0, 0), (width, 42), (12,12,12), -1)
    cv2.putText(panel, "Input RGB", (14,29), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (245,245,245), 2, cv2.LINE_AA)
    cv2.putText(panel, f"frame {idx:06d}", (14,height-16), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255,255,255), 1, cv2.LINE_AA)
    return panel


def render_mp4(output, frames, models, fps):
    first = cv2.imread(str(frames[0]))
    cell_w, cell_h = 640, 420
    target = output / "comparison.mp4"
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (cell_w*2, cell_h*2))
    names = [n for n in ("batrack", "droid_w", "wildpose") if n in models]
    for idx, path in enumerate(frames):
        frame = cv2.imread(str(path))
        panels = [draw_source(frame, idx, cell_w, cell_h)]
        for slot in range(3):
            if slot < len(names):
                name = names[slot]
                panels.append(draw_panel(frame, name, models[name]["poses"], idx, cell_w, cell_h))
            else:
                blank = np.full((cell_h,cell_w,3), 20,np.uint8)
                cv2.putText(blank, "Model output unavailable", (120,210), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (160,160,160), 1, cv2.LINE_AA)
                panels.append(blank)
        writer.write(np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])]))
    writer.release()
    # OpenCV's mp4v is broadly decodable; transcode to browser/PPT-friendly H.264 when available.
    temp = output / "comparison.mp4v.mp4"
    target.replace(temp)
    import subprocess
    result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                             "-threads", "2", "-filter_threads", "2", "-i", str(temp),
                             "-c:v", "libx264", "-preset", "veryfast", "-threads", "2",
                             "-crf", "20", "-pix_fmt", "yuv420p", str(target)])
    if result.returncode == 0:
        temp.unlink()
    else:
        temp.replace(target)


def trajectory_figure(output, models):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    views = [(0,1,"X","Y"), (0,2,"X","Z"), (1,2,"Y","Z")]
    for ax, (a,b,xl,yl) in zip(axes, views):
        for name, item in models.items():
            p = item["poses"][:, :3, 3]
            ax.plot(p[:,a], p[:,b], color=np.array(COLORS_RGB[name])/255, label=LABELS[name], lw=2)
            ax.scatter(p[0,a], p[0,b], marker="o", s=35, color=np.array(COLORS_RGB[name])/255)
            ax.scatter(p[-1,a], p[-1,b], marker="x", s=45, color=np.array(COLORS_RGB[name])/255)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=.25)
    axes[0].legend()
    fig.suptitle("Camera trajectories — each method origin/monocular scale normalized")
    fig.tight_layout()
    fig.savefig(output / "trajectory_comparison.png", dpi=180)
    plt.close(fig)


def create_rrd(output, frames, models, intrinsics):
    """Trajectory-only recording; scenes are exported separately."""
    import rerun as rr
    import rerun.blueprint as rrb
    blueprint = rrb.Blueprint(rrb.Grid(
        rrb.Spatial2DView(name="Input RGB", origin="/input"),
        rrb.Spatial3DView(name="Camera trajectories", origin="/world"),
        rrb.TimeSeriesView(name="Normalized camera step", origin="/metrics"),
        grid_columns=2), collapse_panels=True)
    rec = new_recording(output / "comparison.rrd", "dynamic_ba_trajectories_v2", blueprint)
    try:
        rec.log("/world", rr.ViewCoordinates.RDF, static=True)
        for name, item in models.items():
            xyz = item["poses"][:, :3, 3].astype(np.float32)
            rec.log(f"/world/{name}/trajectory", rr.LineStrips3D([xyz], colors=[COLORS_RGB[name]]), static=True)
            rec.log(f"/metrics/{name}/step", rr.SeriesLines(colors=[COLORS_RGB[name]], names=LABELS[name]), static=True)
        for idx, path in enumerate(frames):
            rec.set_time("frame", sequence=idx)
            rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            rec.log("/input/rgb", rr.Image(rgb).compress(jpeg_quality=90))
            for name, item in models.items():
                pose_idx = min(idx, len(item["poses"])-1)
                pose = item["poses"][pose_idx]
                xyz = pose[:3, 3]
                rec.log(f"/world/{name}/current_camera", rr.Points3D([xyz], colors=[COLORS_RGB[name]], radii=[0.12]))
                log_camera(rec, f"/world/{name}", pose, intrinsics, rgb.shape[1::-1], COLORS_RGB[name])
                if pose_idx:
                    prev = item["poses"][pose_idx-1, :3, 3]
                    rec.log(f"/metrics/{name}/step", rr.Scalars([float(np.linalg.norm(xyz-prev))]))
        rec.flush()
    finally:
        rec.disconnect()


def write_scene_data(output, models, frame_count):
    data = {"frame_count": frame_count, "models": {}}
    for name, item in models.items():
        data["models"][name] = {
            "label": LABELS[name], "color": [v/255 for v in COLORS_RGB[name]],
            "positions": item["poses"][:, :3, 3].tolist(),
            "rotations": item["poses"][:, :3, :3].tolist(),
        }
    (output / "scene_data.json").write_text(json.dumps(data), encoding="utf-8")


def cmd_write_intrinsics(args):
    args.directory.mkdir(parents=True, exist_ok=True)
    fx,fy,cx,cy = args.values
    k = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float32)
    for i in range(args.count):
        np.save(args.directory / f"frame{i:06d}_intrinsics.npy", k)


def cmd_render(args):
    if args.scene_max_frames < 0 or args.scene_points_per_frame < 1:
        raise ValueError("scene frame limit must be >=0 and point limit must be positive")
    if not np.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("FPS must be finite and positive")
    if not np.isfinite(args.scene_far_ratio) or args.scene_far_ratio < 0:
        raise ValueError("scene far ratio must be finite and nonnegative")
    output = args.output.resolve()
    frames = sorted((output / "input/frames").glob("frame*.png"))
    if not frames:
        raise RuntimeError("prepared input frames are missing")
    models = load_models(output, args.models)
    if not args.rrd_only:
        print("Rendering trajectory diagnostics...", flush=True)
        trajectory_figure(output, models)
        render_mp4(output, frames, models, args.fps)
    create_rrd(output, frames, models, args.intrinsics)
    print("Loading saved scene depths and exporting dynamic RRDs...", flush=True)
    scenes, failures = load_scenes(output, frames, models, args.intrinsics,
                                  args.scene_max_frames, args.scene_points_per_frame, args.scene_far_ratio)
    export_scenes(output, frames, scenes, LABELS, COLORS_RGB, failures, {
        "fps": args.fps,
        "max_frames": args.scene_max_frames, "points_per_frame": args.scene_points_per_frame,
        "far_ratio": args.scene_far_ratio,
        "geometry": "depth-derived point clouds; no smoothing, fusion or dynamic-object removal",
        "depth_units": "same median-step normalization as camera translation; not meters",
    })
    write_scene_data(output, models, len(frames))
    summary = {
        "warning": "No GT: visualizations diagnose consistency/failure but do not prove absolute accuracy.",
        "point_cloud_warning": "Rerun point clouds are depth-derived and reflect both depth-prior and BA quality.",
        "trajectory_normalization": "first pose identity; translation scaled by inverse median frame step",
        "models": {n: {"trajectory": x["path"], "frames": len(x["poses"]),
                       "normalization_scale": x["scale"]} for n,x in models.items()},
    }
    (output / "diagnostics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.rrd_only:
        from scene_video import render_scene_video
        render_scene_video(output / "scene_comparison.rrd", output / "scene_dynamic_preview.mp4",
                           args.fps, len(frames), *args.preview_size, renderer=args.preview_renderer)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("write-intrinsics")
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--values", type=float, nargs=4, required=True)
    p.set_defaults(func=cmd_write_intrinsics)
    p = sub.add_parser("render")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fps", type=float, required=True)
    p.add_argument("--intrinsics", type=float, nargs=4, required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--rrd-only", action="store_true", help="rebuild RRDs without rendering MP4/PNG")
    p.add_argument("--scene-max-frames", type=int, default=0, help="maximum sampled depth frames per model; 0 uses all")
    p.add_argument("--preview-size", type=int, nargs=2, default=[1400, 900], metavar=("WIDTH", "HEIGHT"))
    p.add_argument("--preview-renderer", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--scene-points-per-frame", type=int, default=12000, help="maximum points per sampled frame")
    p.add_argument("--scene-far-ratio", type=float, default=10.0,
                   help="fixed far cutoff = ratio * first valid depth median; 0 disables clipping")
    p.set_defaults(func=cmd_render)
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
