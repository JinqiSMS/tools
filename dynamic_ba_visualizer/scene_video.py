"""Render the actual dynamic Rerun scene to a constant-frame-rate MP4."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import queue
import shutil
import socket
import subprocess
import tempfile
import threading
import time


def stop(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


class ViewerClient:
    """Line-delimited MCP client with bounded reads, including server crashes."""
    def __init__(self, executable, env, log):
        self.proc = subprocess.Popen([str(executable), "viewer-mcp"], env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True)
        self.messages = queue.Queue()
        self.seq = 0
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        try:
            for line in self.proc.stdout:
                self.messages.put(json.loads(line))
        except Exception as exc:
            self.messages.put(exc)
        finally:
            self.messages.put(EOFError("Rerun MCP process closed its output"))

    def request(self, method, params, timeout=45):
        self.seq += 1
        self.proc.stdin.write(json.dumps(dict(jsonrpc="2.0", id=self.seq,
                                              method=method, params=params)) + "\n")
        self.proc.stdin.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                message = self.messages.get(timeout=max(0, deadline-time.monotonic()))
            except queue.Empty:
                raise TimeoutError(f"Rerun MCP timed out: {method}") from None
            if isinstance(message, Exception):
                raise message
            if message.get("id") != self.seq:
                continue
            if "error" in message:
                raise RuntimeError(str(message["error"]))
            result = message["result"]
            if result.get("isError"):
                raise RuntimeError(str(result.get("content"))[:2000])
            return result

    def call(self, name, **arguments):
        result = self.request("tools/call", dict(name=name, arguments=arguments))
        for item in result.get("content", []):
            if item.get("type") == "text":
                try:
                    return json.loads(item["text"])
                except ValueError:
                    continue
        return result

    def initialize(self):
        self.request("initialize", dict(protocolVersion="2024-11-05", capabilities={},
            clientInfo=dict(name="dynamic-scene-export", version="1")))
        self.proc.stdin.write(json.dumps(dict(jsonrpc="2.0",
                            method="notifications/initialized")) + "\n")
        self.proc.stdin.flush()


def viewer_executable():
    import rerun
    # Use the native binary so cleanup terminates the actual server, not a wrapper.
    binary = Path(rerun.__file__).resolve().parent.parent / "rerun_cli/rerun"
    if not binary.is_file():
        raise RuntimeError(f"Rerun native viewer not found: {binary}")
    return binary


def render_scene_video(rrd, destination, fps, frame_count, width=1400, height=900,
                       renderer="cpu"):
    if not math.isfinite(fps) or fps <= 0 or frame_count < 1:
        raise ValueError("FPS must be finite and positive; frame count must be positive")
    if min(width, height) < 320 or width % 2 or height % 2:
        raise ValueError("video width/height must be even and at least 320")
    rrd, destination = Path(rrd).resolve(), Path(destination).resolve()
    if not rrd.is_file():
        raise FileNotFoundError(rrd)
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise RuntimeError(f"{tool} is required")
    executable = viewer_executable()
    env = os.environ.copy()
    env.update(RUST_LOG="error", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    if renderer == "cpu":
        drivers = sorted(Path("/usr/share/vulkan/icd.d").glob("lvp_icd*.json"))
        if not drivers:
            raise RuntimeError("CPU rendering requires Mesa lavapipe Vulkan; alternatively use --preview-renderer gpu")
        env["VK_ICD_FILENAMES"] = str(drivers[0])
        env["VK_DRIVER_FILES"] = str(drivers[0])
    destination.parent.mkdir(parents=True, exist_ok=True)
    logs = destination.parent / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / "scene_video.log"
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    viewer = client = None
    started = time.monotonic()
    with log_path.open("w") as log, tempfile.TemporaryDirectory(
            prefix=".scene-video-", dir=destination.parent) as work:
        work = Path(work)
        try:
            viewer = subprocess.Popen([str(executable), "--headless", "--renderer", "vulkan",
                "--bind", "127.0.0.1", "--port", str(port), "--window-size",
                f"{width}x{height}", str(rrd)], env=env, stdout=log, stderr=log)
            deadline = time.monotonic() + 60
            while True:
                if viewer.poll() is not None:
                    raise RuntimeError(f"Rerun viewer exited; see {log_path}")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Rerun startup timed out; see {log_path}")
                    time.sleep(.2)
            client = ViewerClient(executable, env, log)
            client.initialize()
            client.call("connect", endpoint=f"http://127.0.0.1:{port}")
            while True:
                state = client.call("viewer_state")
                if state.get("active_store_id"):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Rerun did not load {rrd}; see {log_path}")
                time.sleep(.2)
            for index in range(frame_count):
                client.call("set_time", time=index, timeline="frame", play=False)
                client.call("wait_for", min_steps=3, timeout_secs=10)
                path = work / f"frame{index:06d}.png"
                client.call("screenshot", save_path=str(path), pixels_per_point=1.0)
                if not path.is_file():
                    raise RuntimeError(f"Rerun did not save screenshot {index}; see {log_path}")
                if index % 10 == 0 or index == frame_count-1:
                    print(f"Dynamic scene video: {index+1}/{frame_count} frames", flush=True)
        finally:
            if client is not None:
                stop(client.proc)
            stop(viewer)
        encoded = work / "preview.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-threads", "2", "-filter_threads", "2", "-framerate", str(fps),
            "-i", str(work / "frame%06d.png"), "-c:v", "libx264", "-preset", "veryfast",
            "-threads", "2", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags",
            "+faststart", str(encoded)], check=True)
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error",
            "-select_streams", "v:0", "-show_entries",
            "stream=width,height,nb_frames,duration,avg_frame_rate", "-of", "json",
            str(encoded)], text=True))["streams"][0]
        if int(probe["nb_frames"]) != frame_count or (probe["width"], probe["height"]) != (width, height):
            raise RuntimeError(f"Unexpected encoded scene video: {probe}")
        encoded.replace(destination)
        report = dict(rrd=str(rrd), video=str(destination), fps=fps, frame_count=frame_count,
            renderer=renderer, probe=probe, seconds=time.monotonic()-started,
            missing_depth="Hold the latest available model depth frame; no interpolation.")
        destination.with_suffix(".json").write_text(json.dumps(report, indent=2))
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="existing results directory")
    parser.add_argument("--preview-size", nargs=2, type=int, default=[1400, 900])
    parser.add_argument("--preview-renderer", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()
    manifest = json.loads((args.output / "input/manifest.json").read_text())
    render_scene_video(args.output / "scene_comparison.rrd",
        args.output / "scene_dynamic_preview.mp4", manifest["fps"], manifest["frame_count"],
        *args.preview_size, renderer=args.preview_renderer)


if __name__ == "__main__":
    main()
