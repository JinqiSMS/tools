"""Check video extraction and reuse boundaries without running GPU models."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run
from scene_video import ViewerClient, render_scene_video


class PipelineTests(unittest.TestCase):
    def test_video_stride_fps_and_cached_source_guard(self):
        with tempfile.TemporaryDirectory(prefix="scene pipeline ") as directory:
            root = Path(directory)
            video = root / "input video.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                "testsrc2=size=64x48:rate=24", "-frames:v", "24", "-c:v", "libx264",
                "-threads", "2", str(video)], check=True)
            args = argparse.Namespace(input=video, resume=False, stride=2, max_frames=-1, fps=None)
            output = root / "results with spaces"
            meta = run.prepare_input(args, output)
            self.assertEqual(len(meta["frames"]), 12)
            self.assertEqual(meta["fps"], 12.)
            self.assertEqual((meta["width"], meta["height"]), (64, 48))
            self.assertEqual(len(list((output / "input/droid_frames").glob("*.jpg"))), 12)
            self.assertTrue(run.cached_input(output, video)["reused"])
            with self.assertRaisesRegex(ValueError, "different input"):
                run.cached_input(output, root / "different video.mp4")
            meta["frames"][-1].unlink()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                run.cached_input(output, video)

    def test_visualize_only_preserves_model_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "input/frames").mkdir(parents=True)
            (output / "input/frames/frame000000.png").touch()
            source = output / "original.mp4"
            (output / "input/manifest.json").write_text(json.dumps(dict(
                source=str(source), frame_count=1, fps=15)))
            status = '{"input": {"intrinsics": [1, 1, 0, 0]}, "models": {"existing": "keep"}}'
            (output / "run_status.json").write_text(status)
            with patch("sys.argv", ["run.py", "--input", str(source), "--output", str(output),
                                    "--visualize-only"]), patch.object(run, "render_outputs") as render:
                self.assertEqual(run.main(), 0)
                self.assertEqual(render.call_args.args[2], [1, 1, 0, 0])
            self.assertEqual((output / "run_status.json").read_text(), status)

    def test_invalid_video_settings_fail_before_viewer(self):
        for fps in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                render_scene_video("missing.rrd", "unused.mp4", fps, 5)

    def test_mcp_timeout_and_closed_server_are_bounded(self):
        import queue
        from unittest.mock import Mock
        client = ViewerClient.__new__(ViewerClient)
        client.proc = Mock()
        client.seq = 0
        client.messages = queue.Queue()
        with self.assertRaises(TimeoutError):
            client.request("test", {}, timeout=.01)
        client.messages.put(EOFError("closed"))
        with self.assertRaises(EOFError):
            client.request("test", {})


if __name__ == "__main__":
    unittest.main()
