#!/usr/bin/env python3
"""Run the original BA-Track, DROID-W and WildPose models on one RGB sequence.

The coordinator intentionally uses only the Python standard library.  Each model
is launched in its existing conda environment and the renderer is launched in
the WildPose environment (which already contains OpenCV, NumPy and Rerun).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


ROOT = Path("/data/tjq")
CONDA = ROOT / "miniconda3/bin/conda"
HERE = Path(__file__).resolve().parent
REPOS = {
    "batrack": ROOT / "batrack",
    "droid_w": ROOT / "DROID-W",
    "wildpose": ROOT / "WildPose",
}
ENVS = {"batrack": "batrack", "droid_w": "droid-w", "wildpose": "wildpose"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def natural_key(path: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def run_logged(command, cwd: Path, log_path: Path, env=None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(str(x) for x in command)
    print(f"\n$ {printable}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {printable}\n")
        proc = subprocess.Popen(
            [str(x) for x in command], cwd=str(cwd), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = proc.wait()
    if code:
        raise RuntimeError(f"command exited with status {code}; see {log_path}")


def conda_command(env_name: str, *args: str):
    return [CONDA, "run", "--no-capture-output", "-n", env_name, *args]


def ffprobe(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate",
        "-of", "json", str(path),
    ]
    data = json.loads(subprocess.check_output(cmd, text=True))
    if not data.get("streams"):
        raise ValueError(f"not a readable image/video: {path}")
    return data["streams"][0]


def parse_rate(value: str | None, fallback: float) -> float:
    if not value or value == "0/0":
        return fallback
    a, b = value.split("/")
    return float(a) / float(b)


def list_images(directory: Path) -> list[Path]:
    images = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(images, key=natural_key)


def prepare_input(args, output: Path) -> dict:
    source = args.input.resolve()
    prep = output / "input"
    frames = prep / "frames"
    droid_frames = prep / "droid_frames"
    wild_rgb = prep / "wildpose_sequence" / "rgb"
    batrack_frames = prep / "batrack_data" / "sequence"
    for path in (frames, droid_frames, wild_rgb, batrack_frames):
        path.mkdir(parents=True, exist_ok=True)

    existing = sorted(frames.glob("frame*.png"), key=natural_key)
    if args.resume and existing:
        if len(existing) < 12:
            raise ValueError(f"cached input has only {len(existing)} frames; remove it or provide 12+")
        for idx, src in enumerate(existing):
            wp = wild_rgb / f"frame{idx:06d}.png"
            bp = batrack_frames / f"frame{idx:06d}.png"
            if not wp.exists(): wp.symlink_to(src)
            if not bp.exists(): bp.symlink_to(src)
            jpg = droid_frames / f"frame{idx:06d}.jpg"
            if not jpg.exists():
                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                     "-threads", "2", "-filter_threads", "2", "-i", str(src),
                     "-q:v", "1", str(jpg)], check=True,
                )
        probe = ffprobe(existing[0])
        manifest_path = prep / "manifest.json"
        old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        fps = float(old.get("fps", args.fps or 10.0))
        return {"frames": existing, "width": int(probe["width"]), "height": int(probe["height"]),
                "fps": fps, "source": str(source), "reused": True}

    for folder in (frames, droid_frames, wild_rgb, batrack_frames):
        for old in folder.iterdir():
            if old.is_file() or old.is_symlink():
                old.unlink()

    if source.is_dir():
        inputs = list_images(source)[:: args.stride]
        if args.max_frames > 0:
            inputs = inputs[: args.max_frames]
        if not inputs:
            raise ValueError(f"no supported images directly under {source}")
        fps = args.fps or 10.0
        for idx, image in enumerate(inputs):
            dst = frames / f"frame{idx:06d}.png"
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                     "-threads", "2", "-filter_threads", "2", "-i", str(image),
                 "-frames:v", "1", "-pix_fmt", "rgb24", str(dst)], check=True,
            )
    elif source.is_file() and source.suffix.lower() in VIDEO_EXTS:
        info = ffprobe(source)
        fps = args.fps or parse_rate(info.get("avg_frame_rate"), 30.0) / args.stride
        vf = f"select=not(mod(n\\,{args.stride}))"
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                     "-threads", "2", "-filter_threads", "2", "-i", str(source),
               "-vf", vf, "-vsync", "0"]
        if args.max_frames > 0:
            cmd += ["-frames:v", str(args.max_frames)]
        cmd += ["-start_number", "0", str(frames / "frame%06d.png")]
        subprocess.run(cmd, check=True)
    elif source.is_file() and source.suffix.lower() in IMAGE_EXTS:
        raise ValueError("a single image is not a temporal sequence; provide a video or image directory")
    else:
        raise ValueError(f"unsupported input: {source}")

    prepared = sorted(frames.glob("frame*.png"), key=natural_key)
    if len(prepared) < 12:
        raise ValueError(f"only {len(prepared)} frames found; these SLAM/BA models require at least 12, preferably 30+")
    probe = ffprobe(prepared[0])
    width, height = int(probe["width"]), int(probe["height"])

    # DROID-W's stock RGB_NoPose loader reads frame*.jpg; WildPose's stock
    # Sintel loader reads rgb/*.png. Keep a lossless master and adapt by links/JPEG.
    for idx, src in enumerate(prepared):
        wp = wild_rgb / f"frame{idx:06d}.png"
        if not wp.exists():
            wp.symlink_to(src)
        bp = batrack_frames / f"frame{idx:06d}.png"
        if not bp.exists():
            bp.symlink_to(src)
        jpg = droid_frames / f"frame{idx:06d}.jpg"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                     "-threads", "2", "-filter_threads", "2", "-i", str(src),
             "-q:v", "1", str(jpg)], check=True,
        )

    manifest = {
        "source": str(source), "frame_count": len(prepared), "width": width, "height": height,
        "fps": fps, "stride": args.stride, "max_frames": args.max_frames,
        "files": [p.name for p in prepared],
    }
    (prep / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"frames": prepared, "width": width, "height": height, "fps": fps,
            "source": str(source), "reused": False}


def tracking_size(
    width: int,
    height: int,
    long_side: int = 640,
    alignment: int = 8,
) -> tuple[int, int]:
    """Choose an aligned tracking size while keeping the source aspect ratio.

    Independently rounding a scaled width and height works for DROID-W's
    8-pixel constraint, but produced 640x360 for WildPose. MASt3R embeds
    16x16 patches, so that height fails before the first frame. Searching the
    small set of aligned candidates avoids both that failure and unnecessary
    aspect-ratio distortion (852x480 becomes 624x352 for alignment=16).
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {width}x{height}")
    target_long = min(long_side, max(width, height))
    aspect = width / height
    candidates = []
    for w in range(64, target_long + 1, alignment):
        for h in range(64, target_long + 1, alignment):
            if max(w, h) > target_long:
                continue
            aspect_error = abs(math.log((w / h) / aspect))
            size_loss = 1.0 - max(w, h) / target_long
            score = 10.0 * aspect_error + 0.02 * size_loss
            candidates.append((aspect_error, score, w, h))
    if not candidates:
        raise ValueError(f"cannot construct an aligned size for {width}x{height}")
    near_aspect = [item for item in candidates if item[0] <= 0.005]
    if near_aspect:
        best = max(near_aspect, key=lambda item: (max(item[2], item[3]), item[2] * item[3]))
    else:
        best = min(candidates, key=lambda item: (item[1], -item[2] * item[3]))
    return best[2], best[3]


def intrinsics(args, width: int, height: int) -> tuple[list[float], str]:
    if args.intrinsics:
        return [float(v) for v in args.intrinsics], "provided"
    f = 0.5 * width / math.tan(math.radians(args.fov_deg) / 2.0)
    return [f, f, (width - 1) / 2.0, (height - 1) / 2.0], f"estimated_from_{args.fov_deg:g}deg_horizontal_fov"


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_configs(output: Path, meta: dict, k: list[float], args) -> dict[str, Path]:
    cfg_dir = output / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    width, height = meta["width"], meta["height"]
    droid_w, droid_h = tracking_size(width, height, alignment=8)
    wild_w, wild_h = tracking_size(width, height, alignment=16)
    fx, fy, cx, cy = k

    droid = cfg_dir / "droid_w.yaml"
    droid.write_text(f"""inherit_from: {REPOS['droid_w'] / 'configs/droid_w.yaml'}
scene: sequence
dataset: youtube
stride: 1
max_frames: -1
gui: false
droidvis: false
mapping:
  enable: false
  eval_before_final_ba: true
data:
  input_folder: {output / 'input/droid_frames'}
  output: {output / 'models/droid_w'}
cam:
  H: {height}
  W: {width}
  fx: {fx:.9f}
  fy: {fy:.9f}
  cx: {cx:.9f}
  cy: {cy:.9f}
  H_edge: 0
  W_edge: 0
  H_out: {droid_h}
  W_out: {droid_w}
  png_depth_scale: 1.0
  distortion: [0, 0, 0, 0, 0]
tracking:
  buffer: {args.buffer}
  backend:
    final_ba: true
  uncertainty_params:
    visualize: true
""", encoding="utf-8")

    wild = cfg_dir / "wildpose.yaml"
    wild.write_text(f"""inherit_from: {REPOS['wildpose'] / 'configs/wildgs_slam.yaml'}
scene: sequence
dataset: sintel
stride: 1
max_frames: -1
delete_mono_priors: false
data:
  input_folder: {output / 'input/wildpose_sequence'}
  output: {output / 'models/wildpose'}
cam:
  H: {height}
  W: {width}
  fx: {fx:.9f}
  fy: {fy:.9f}
  cx: {cx:.9f}
  cy: {cy:.9f}
  H_edge: 0
  W_edge: 0
  H_out: {wild_h}
  W_out: {wild_w}
  png_depth_scale: 1.0
tracking:
  buffer: {args.buffer}
  full_ba: false
  backend:
    final_ba: true
""", encoding="utf-8")
    return {"droid_w": droid, "wildpose": wild}


def run_batrack(output: Path, args, k: list[float]) -> str:
    repo = REPOS["batrack"]
    frames = output / "input/batrack_data/sequence"
    depth_root = output / "cache/batrack_depth"
    scene = "sequence"
    da = depth_root / "depthAny_disp" / scene
    uni = depth_root / "unidepthv2" / scene
    aligned = depth_root / "unidepth_dav2" / scene
    calib = depth_root / "unidepth_dav2_intrinsics" / scene
    for p in (da, uni, aligned, calib):
        p.mkdir(parents=True, exist_ok=True)
    log = output / "logs/batrack.log"
    expected = len(list(frames.glob("frame*.png")))
    if not (args.resume and len(list(da.glob("*.npy"))) == expected):
        run_logged(conda_command("batrack", "python", "Depth-Anything/run_videos_v2.py",
                                 "--encoder", "vitl", "--load-from",
                                 "Depth-Anything/checkpoints/depth_anything_v2_vitl.pth",
                                 "--img-path", str(frames), "--outdir", str(da)), repo, log)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "UniDepth") + os.pathsep + env.get("PYTHONPATH", "")
    if not (args.resume and len(list(uni.glob("*.npz"))) == expected):
        run_logged(conda_command("batrack", "python", "UniDepth/scripts/demo_mega-sam.py",
                                 "--scene-name", scene, "--img-path", str(frames),
                                 "--outdir", str(depth_root / "unidepthv2")), repo, log, env=env)
    if not (args.resume and len(list(aligned.glob("*.npy"))) == expected):
        run_logged(conda_command("batrack", "python", "main/mono_depth/get_mono_depth.py",
                                 "--data_dir", str(output / "input/batrack_data"), "--depth_dir", str(depth_root),
                                 "--save_name", "unidepth_dav2"), repo, log)
    if args.intrinsics:
        run_logged(conda_command("wildpose", "python", str(HERE / "visualize.py"),
                                 "write-intrinsics", "--directory", str(calib), "--count", str(expected),
                                 "--values", *[str(x) for x in k]), ROOT, log)
    traj = output / "models/batrack/sequence/batrack_traj.txt"
    if args.resume and traj.exists():
        print(f"Reusing {traj}")
        return "reused"
    overrides = [
        f"data.imagedir={frames}", f"data.savedir={output / 'models/batrack'}",
        f"+data.depthdir={aligned}", f"+data.depthdir_gt={aligned}", f"data.calib={calib}",
        "data.name=sequence", "data.traj_format=davis", "+data.input_intrinsics=true",
        "save_trajectory=true", "save_plot=false", "save_results=false", "save_video=false",
        "model.motion_decomposition_space=uvd",
    ]
    if args.low_memory:
        overrides += ["slam.PATCHES_PER_FRAME=64", "slam.PATCH_GEN=grid_grad_8"]
    cmd = conda_command("batrack", "python", "main/run_batrack.py",
                        f"--config-path={repo / 'configs'}", "--config-name=davis_demo", *overrides)
    run_logged(cmd, repo, log)
    return "ran"


def run_droid(output: Path, config: Path, args) -> str:
    traj = output / "models/droid_w/sequence/traj/est_poses_full.txt"
    if args.resume and traj.exists():
        print(f"Reusing {traj}")
        return "reused"
    run_logged(conda_command("droid-w", "python", "run.py", "--config", str(config)),
               REPOS["droid_w"], output / "logs/droid_w.log")
    return "ran"


def run_wildpose(output: Path, config: Path, args) -> str:
    traj = output / "models/wildpose/sequence/traj/est_poses_full.txt"
    if args.resume and traj.exists():
        print(f"Reusing {traj}")
        return "reused"
    run_logged(conda_command("wildpose", "python", "run.py", str(config)),
               REPOS["wildpose"], output / "logs/wildpose.log")
    return "ran"


def render_outputs(output, meta, k, models, args):
    command = conda_command(
        "wildpose", "python", str(HERE / "visualize.py"), "render",
        "--output", str(output), "--fps", str(meta["fps"]),
        "--intrinsics", *[str(x) for x in k], "--models", *models,
        "--scene-max-frames", str(args.scene_max_frames),
        "--scene-points-per-frame", str(args.scene_points_per_frame),
        "--scene-far-ratio", str(args.scene_far_ratio),
        "--preview-size", *map(str, args.preview_size), "--preview-renderer", args.preview_renderer,
    )
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    run_logged(command, ROOT, output / "logs/visualization.log", env=env)
    print(f"Dynamic scene video: {output / 'scene_dynamic_preview.mp4'}")
    print(f"Dynamic scenes: {output / 'scene_comparison.rrd'} (individual files in {output / 'scenes'})")


def cached_input(output, source):
    manifest = json.loads((output / "input/manifest.json").read_text())
    if Path(manifest["source"]).resolve() != source.resolve():
        raise ValueError("output contains a different input; use a different --output directory")
    frames = sorted((output / "input/frames").glob("frame*.png"), key=natural_key)
    if len(frames) != manifest["frame_count"]:
        raise ValueError("cached input frames are incomplete")
    return {**manifest, "frames": frames, "reused": True}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run and visualize original BA-Track, DROID-W and WildPose on RGB video/images.")
    parser.add_argument("--input", type=Path, required=True, help="video file or directory of ordered images")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", default="batrack,droid_w,wildpose",
                        help="comma-separated: batrack,droid_w,wildpose")
    parser.add_argument("--intrinsics", nargs=4, type=float, metavar=("FX", "FY", "CX", "CY"))
    parser.add_argument("--fov-deg", type=float, default=60.0,
                        help="fallback horizontal FOV when intrinsics are unknown")
    parser.add_argument("--fps", type=float, help="output FPS; directory input defaults to 10")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--buffer", type=int, default=300, help="DROID-W/WildPose keyframe buffer")
    parser.add_argument("--low-memory", action="store_true", help="BA-Track 64-patch fallback")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-going", action="store_true", help="render successful models if one fails")
    parser.add_argument("--visualize-only", action="store_true", help="reuse existing model outputs")
    parser.add_argument("--blender", type=Path, help="optional Blender executable")
    parser.add_argument("--scene-max-frames", type=int, default=0,
                        help="depth frames per model in scene exports; 0 uses all saved frames")
    parser.add_argument("--scene-points-per-frame", type=int, default=12000)
    parser.add_argument("--scene-far-ratio", type=float, default=10.0)
    parser.add_argument("--preview-size", type=int, nargs=2, default=[1400, 900], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--preview-renderer", choices=["cpu", "gpu"], default="cpu",
                        help="headless scene video renderer; CPU avoids additional GPU memory use")
    args = parser.parse_args()
    if args.stride < 1 or args.buffer < 16:
        parser.error("--stride must be >=1 and --buffer must be >=16")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps must be finite and positive")
    if not 0 < args.fov_deg < 180:
        parser.error("--fov-deg must be between 0 and 180")
    if args.scene_max_frames < 0 or args.scene_points_per_frame < 1:
        parser.error("--scene-max-frames must be >=0 and --scene-points-per-frame must be positive")
    if not math.isfinite(args.scene_far_ratio) or args.scene_far_ratio < 0:
        parser.error("--scene-far-ratio must be finite and >=0")
    if any(x < 320 or x % 2 for x in args.preview_size):
        parser.error("--preview-size dimensions must be even and >=320")
    aliases = {"droid-w": "droid_w", "ba-track": "batrack", "wild-pose": "wildpose"}
    models = [aliases.get(x.strip().lower(), x.strip().lower()) for x in args.models.split(",") if x.strip()]
    unknown = set(models) - set(REPOS)
    if not models:
        parser.error("at least one model is required")
    if unknown:
        parser.error(f"unknown models: {sorted(unknown)}")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.visualize_only:
        meta = cached_input(output, args.input)
        previous = json.loads((output / "run_status.json").read_text())
        if args.intrinsics or args.fps is not None:
            parser.error("--visualize-only uses recorded calibration and FPS; omit --intrinsics and --fps")
        render_outputs(output, meta, previous["input"]["intrinsics"], models, args)
        return 0
    if (output / "input/manifest.json").exists():
        if not args.resume:
            parser.error("output already contains a run; use --resume, --visualize-only or a new --output")
        old = cached_input(output, args.input)
        if old["stride"] != args.stride or ("max_frames" in old and old["max_frames"] != args.max_frames):
            parser.error("cached extraction differs from --stride/--max-frames; use a new --output")
    meta = prepare_input(args, output)
    k, k_source = intrinsics(args, meta["width"], meta["height"])
    if args.resume and (output / "run_status.json").exists():
        previous = json.loads((output / "run_status.json").read_text())["input"]
        if args.intrinsics and list(args.intrinsics) != previous["intrinsics"]:
            parser.error("cached models use different intrinsics; use a new --output")
        if args.fps is not None and args.fps != meta["fps"]:
            parser.error("cached input uses a different FPS; use a new --output")
        k, k_source = previous["intrinsics"], previous["intrinsics_source"]
    configs = write_configs(output, meta, k, args)
    status = {
        "input": {**{x: meta[x] for x in ("source", "width", "height", "fps")},
                  "frame_count": len(meta["frames"]), "intrinsics": k, "intrinsics_source": k_source},
        "models": {}, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "weights": {
            "batrack": {"path": str(REPOS['batrack'] / 'checkpoints/md_tracker.pth'),
                        "sha256": sha256(REPOS['batrack'] / 'checkpoints/md_tracker.pth')},
            "droid_w": {"path": str(REPOS['droid_w'] / 'pretrained/droid.pth'),
                        "sha256": sha256(REPOS['droid_w'] / 'pretrained/droid.pth')},
            "wildpose": {"path": str(REPOS['wildpose'] / 'pretrained/wildpose_v0.pth'),
                         "sha256": sha256(REPOS['wildpose'] / 'pretrained/wildpose_v0.pth')},
        },
        "repositories": {},
    }
    for name, repo in REPOS.items():
        try:
            commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            dirty = bool(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip())
            status["repositories"][name] = {"path": str(repo), "commit": commit, "dirty": dirty}
        except Exception as exc:
            status["repositories"][name] = {"path": str(repo), "error": str(exc)}
    status_path = output / "run_status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    runners = {
        "batrack": lambda: run_batrack(output, args, k),
        "droid_w": lambda: run_droid(output, configs["droid_w"], args),
        "wildpose": lambda: run_wildpose(output, configs["wildpose"], args),
    }
    if not args.visualize_only:
        for model in models:
            started = time.time()
            try:
                mode = runners[model]()
                status["models"][model] = {"status": "ok", "mode": mode,
                                           "seconds_this_invocation": time.time() - started}
            except Exception as exc:
                status["models"][model] = {"status": "failed", "seconds": time.time() - started,
                                           "error": str(exc)}
                status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
                if not args.keep_going:
                    raise
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    render_models = [name for name in models if status["models"][name]["status"] == "ok"]
    if not render_models:
        raise RuntimeError("No models completed successfully; see run_status.json and logs")
    render_outputs(output, meta, k, render_models, args)

    blender = args.blender or (Path(shutil.which("blender")) if shutil.which("blender") else None)
    if blender:
        blender_dir = output / "blender"
        blender_dir.mkdir(exist_ok=True)
        cmd = [blender, "--background", "--python", HERE / "blender_scene.py", "--",
               output / "scene_data.json", blender_dir / "comparison.blend",
               blender_dir / "frames", str(meta["fps"])]
        try:
            run_logged(cmd, ROOT, output / "logs/blender.log")
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate",
                            str(meta["fps"]), "-start_number", "1", "-i",
                            str(blender_dir / "frames/frame_%04d.png"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            str(blender_dir / "comparison_blender.mp4")], check=True)
        except Exception as exc:
            print(f"Blender export failed: {exc}", file=sys.stderr)
    else:
        note = output / "blender/README.txt"
        note.parent.mkdir(exist_ok=True)
        note.write_text(
            "Blender executable was not found. scene_data.json is ready. Run:\n"
            f"blender --background --python {HERE / 'blender_scene.py'} -- "
            f"{output / 'scene_data.json'} {output / 'blender/comparison.blend'} "
            f"{output / 'blender/frames'} {meta['fps']}\n", encoding="utf-8")

    status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"\nDone. Trajectories: {output / 'comparison.rrd'}")
    print(f"Scenes: {output / 'scene_comparison.rrd'} (individual views in {output / 'scenes'})")
    print(f"Dynamic scene video: {output / 'scene_dynamic_preview.mp4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
