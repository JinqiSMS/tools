"""Geometry regressions: wrong scales/crops produce misleading scene comparisons."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from scene_export import export_scenes, load_scenes, points_from_depth, prepared_rgb


class SceneGeometryTests(unittest.TestCase):
    def test_default_export_keeps_more_than_sixty_native_depth_frames(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            depth_dir = output / "cache/batrack_depth/unidepth_dav2/sequence"
            depth_dir.mkdir(parents=True)
            frames = []
            for index in range(65):
                path = output / f"frame{index:06d}.png"
                cv2.imwrite(str(path), np.full((16, 16, 3), index, dtype=np.uint8))
                np.save(depth_dir / f"{path.stem}.npy", np.full((16, 16), 1+index/100))
                frames.append(path)
            model = dict(poses=np.repeat(np.eye(4)[None], 65, axis=0), scale=1.)
            scenes, failures = load_scenes(output, frames, {"batrack": model}, [8, 8, 8, 8])
            self.assertFalse(failures)
            self.assertEqual([s["frame"] for s in scenes["batrack"]["snapshots"]], list(range(65)))

    def test_camera_to_world_and_depth_scale_applied_once(self):
        pose = np.eye(4)
        pose[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        pose[:3, 3] = [10, 20, 30]
        rgb = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
        points, colors = points_from_depth(np.array([[2., 2.]]), rgb, pose,
                                           [2, 2, 0, 0], 3, 100)
        np.testing.assert_allclose(points, [[16, 20, 30], [16, 20, 27]])
        np.testing.assert_array_equal(colors, rgb[0])

    def test_invalid_mask_and_far_depth(self):
        depth = np.array([[1, np.nan, np.inf, -1, 0, 10000, 2, 3.]])
        rgb = np.zeros((1, 8, 3), dtype=np.uint8)
        mask = np.ones(depth.shape, dtype=bool)
        mask[0, 6] = False
        points, _ = points_from_depth(depth, rgb, np.eye(4), [1, 1, 0, 0],
                                      1, 100, mask=mask, far_depth=10)
        np.testing.assert_allclose(points, [[0, 0, 1], [21, 0, 3]])

    def test_resize_and_crop_intrinsics(self):
        cam = dict(H=100, W=200, H_out=40, W_out=80, H_edge=5, W_edge=10,
                   fx=120, fy=140, cx=100, cy=50)
        rgb, k = prepared_rgb(np.zeros((100, 200, 3), dtype=np.uint8), cam)
        self.assertEqual(rgb.shape, (40, 80, 3))
        np.testing.assert_allclose(k, [60, 70, 40, 20])

    def test_point_cap_and_empty_cloud(self):
        depth = np.ones((1, 10000))
        rgb = np.zeros((1, 10000, 3), dtype=np.uint8)
        points, _ = points_from_depth(depth, rgb, np.eye(4), [1, 1, 0, 0], 1, 10)
        self.assertLessEqual(len(points), 10)
        points, colors = points_from_depth(depth*0, rgb, np.eye(4), [1, 1, 0, 0], 1, 10)
        self.assertEqual(points.shape, (0, 3))
        self.assertEqual(colors.shape, (0, 3))

    def test_misaligned_rgb_is_rejected(self):
        with self.assertRaises(ValueError):
            points_from_depth(np.ones((4, 4)), np.zeros((8, 8, 3)), np.eye(4),
                              [1, 1, 0, 0], 1, 100)

    def test_exported_geometry_changes_on_sparse_frame_timeline(self):
        """Decode the actual files: a static merged scene must fail this test."""
        from rerun.experimental import RrdReader
        with TemporaryDirectory() as directory:
            output = Path(directory)
            rgb = np.full((4, 4, 3), 128, dtype=np.uint8)
            frames = [output / f"{i}.png" for i in range(4)]
            for path in frames:
                cv2.imwrite(str(path), rgb)
            snapshots = []
            for idx in (0, 2):
                points = np.array([[idx, 0., 2.]], dtype=np.float32)
                color = np.array([[255, 0, 0]], dtype=np.uint8)
                snapshots.append(dict(frame=idx, points=points, colors=color,
                    dynamic_points=points, dynamic_colors=color, k=np.array([2, 2, 2, 2]),
                    rgb=rgb, depth=np.ones((4, 4), dtype=np.float32)))
            scene = dict(snapshots=snapshots, poses=np.repeat(np.eye(4)[None], 4, axis=0),
                         source="test", calibration="test", fallback_intrinsics=False,
                         far_depth=10., scale=1., mask="none", available_frames=2)
            export_scenes(output, frames, {"test": scene}, {"test": "Test"},
                          {"test": (255, 0, 0)}, {}, {"fps": 30.})
            for path, entity in [(output / "scene_comparison.rrd", "/scenes/test/points"),
                                 (output / "scenes/test.rrd", "/world/points")]:
                samples = {}
                for chunk in RrdReader(path).stream():
                    if chunk.entity_path != entity:
                        continue
                    batch = chunk.to_record_batch()
                    self.assertIn("frame", batch.schema.names, "geometry must be temporal, not static")
                    key = next(n for n in batch.schema.names if "positions" in n)
                    samples.update(zip(batch.column("frame").to_pylist(), batch.column(key).to_pylist()))
                self.assertEqual(sorted(samples), [0, 2], "do not fabricate missing depth frames")
                np.testing.assert_allclose(samples[0], [[0, 0, 2]])
                np.testing.assert_allclose(samples[2], [[2, 0, 2]])


if __name__ == "__main__":
    unittest.main()
