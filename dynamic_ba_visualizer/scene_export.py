"""Depth-backed scene recordings, using each model's saved camera geometry."""

from uuid import uuid4
import json
import math

import cv2
import numpy as np
import yaml


def points_from_depth(depth, rgb, pose, intrinsics, scale, max_points,
                      mask=None, far_depth=None):
    """Backproject aligned RGB/depth with K in depth pixels; pose is normalized C2W.

    Only depth is scaled here: the pose translation has already been scaled.
    No smoothing/fusion is applied, so inter-frame misalignment remains visible.
    """
    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2 or rgb.shape[:2] != depth.shape:
        raise ValueError("RGB and depth must have the same H x W")
    fx, fy, cx, cy = np.asarray(intrinsics, dtype=float)
    if not np.isfinite([fx, fy, cx, cy]).all() or min(fx, fy) <= 0:
        raise ValueError("invalid camera intrinsics")
    if max_points < 1:
        raise ValueError("max_points must be positive")
    h, w = depth.shape
    stride = max(1, math.ceil(math.sqrt(h * w / max_points)))
    v, u = np.mgrid[:h:stride, :w:stride]
    z = depth[::stride, ::stride]
    valid = np.isfinite(z) & (z > 1e-5)
    if mask is not None:
        if mask.shape != depth.shape:
            raise ValueError("depth mask shape mismatch")
        valid &= mask[::stride, ::stride].astype(bool)
    if far_depth is not None:
        valid &= z <= far_depth
    z = z[valid].astype(np.float64) * scale
    camera = np.column_stack(((u[valid] - cx) * z / fx,
                              (v[valid] - cy) * z / fy, z))
    world = camera @ pose[:3, :3].T + pose[:3, 3]
    colors = rgb[::stride, ::stride][valid]
    # The ceil stride can still exceed the cap on very narrow images.
    selected = np.linspace(0, len(world)-1, min(len(world), max_points), dtype=int)
    return world[selected].astype(np.float32), colors[selected]


def prepared_rgb(image, cam):
    """Match DROID-W/WildPose dataset resize, undistortion and edge crop."""
    h, w = int(cam["H_out"]), int(cam["W_out"])
    eh, ew = int(cam["H_edge"]), int(cam["W_edge"])
    fx, fy, cx, cy = [float(cam[k]) for k in ("fx", "fy", "cx", "cy")]
    if cam.get("distortion") is not None:
        image = cv2.undistort(image, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]]),
                              np.asarray(cam["distortion"], dtype=float))
    image = cv2.resize(image, (w + 2*ew, h + 2*eh))
    rgb = cv2.cvtColor(image[eh:eh+h, ew:ew+w], cv2.COLOR_BGR2RGB)
    sx, sy = (w + 2*ew) / cam["W"], (h + 2*eh) / cam["H"]
    return rgb, np.array([fx*sx, fy*sy, cx*sx-ew, cy*sy-eh])


def load_scenes(output, frames, models, intrinsics, max_snapshots=0,
                max_points=12000, far_ratio=10.0):
    """Load bounded samples. Archive depths use their paired archive C2W poses."""
    scenes, failures = {}, {}
    for name, model in models.items():
        try:
            scene = _load_scene(output, frames, name, model, intrinsics,
                                max_snapshots, max_points, far_ratio)
            if not scene["snapshots"]:
                raise ValueError("no valid depth points after filtering")
            scenes[name] = scene
        except (OSError, ValueError, KeyError, IndexError) as exc:
            failures[name] = str(exc)
            print(f"warning: scene export unavailable for {name}: {exc}")
    return scenes, failures


def _load_scene(output, frames, name, model, intrinsics, max_snapshots, max_points, far_ratio):
    archive = None
    base = output / "models" / name / "sequence"
    cam = None
    if name == "batrack":
        depth_dir = output / "cache/batrack_depth/unidepth_dav2/sequence"
        paths = [depth_dir / f"{p.stem}.npy" for p in frames]
        if not all(p.exists() for p in paths):
            raise ValueError("BA-Track cached depth frames are missing")
        timestamps = np.arange(min(len(paths), len(model["poses"])))
        source = "BA-Track input depth prior + final camera poses (not optimized dense depth)"
    else:
        archive = np.load(base / "video.npz", allow_pickle=False)
        timestamps_raw = archive["timestamps"]
        timestamps = timestamps_raw.astype(int)
        if not np.array_equal(timestamps, timestamps_raw):
            archive.close()
            raise ValueError("expected integer prepared-frame timestamps")
        source = "Saved optimized keyframe depth + paired keyframe camera poses"
    try:
        if len(timestamps) == 0 or np.any(np.diff(timestamps) <= 0):
            raise ValueError("keyframe timestamps must be nonempty and strictly increasing")
        if timestamps.min() < 0 or timestamps.max() >= min(len(frames), len(model["poses"])):
            raise ValueError("depth timestamp outside the RGB/trajectory range")
        scene_poses = model["poses"].copy()
        if archive is not None:
            with (base / "cfg.yaml").open() as handle:
                cam = yaml.safe_load(handle)["cam"]
            raw_poses = archive["poses"]
            if raw_poses.shape != (len(timestamps), 4, 4):
                raise ValueError("invalid saved keyframe pose shape")
            paired = np.linalg.inv(model["raw"][0]) @ raw_poses
            paired[:, :3, 3] *= model["scale"]
            scene_poses[timestamps] = paired
            # Load NPZ arrays once, rather than repeatedly decompressing them.
            if name == "droid_w":
                disps = archive["droid_disps_up"]
                depths = np.full_like(disps, np.nan)
                np.divide(1.0, disps, out=depths, where=disps > 1e-6)
                rgbs = np.clip(archive["images"].transpose(0, 2, 3, 1)*255, 0, 255).astype(np.uint8)
                low_h, low_w = archive["droid_disps"].shape[-2:]
                high_h, high_w = depths.shape[-2:]
                ks = archive["intrinsics"] * [high_w/low_w, high_h/low_h,
                                              high_w/low_w, high_h/low_h]
                masks = None
                calibration = "video.npz intrinsics rescaled from disparity grid to depth grid"
            else:
                depths, masks = archive["depths"], archive["valid_depth_masks"]
                calibration = "saved cfg.yaml camera with dataset resize and crop"
            if len(depths) != len(timestamps):
                raise ValueError("keyframe depth/timestamp count mismatch")
        else:
            calibration = "per-frame BA-Track cached intrinsics with dataset bottom/right crop"
        count = min(len(timestamps), max_snapshots) if max_snapshots > 0 else len(timestamps)
        selected = np.linspace(0, len(timestamps)-1, count, dtype=int)
        snapshots, far_depth = [], None
        used_fallback_k = False
        for j in selected:
            idx = int(timestamps[j])
            mask = None
            if name == "batrack":
                image = cv2.imread(str(frames[idx]))
                if image is None:
                    raise ValueError(f"unreadable RGB frame {idx}")
                h, w = image.shape[:2]
                h, w = h-h%16, w-w%16
                rgb = cv2.cvtColor(image[:h, :w], cv2.COLOR_BGR2RGB)
                depth = np.load(paths[idx]).squeeze()[:h, :w]
                kp = output / "cache/batrack_depth/unidepth_dav2_intrinsics/sequence" / f"{frames[idx].stem}_intrinsics.npy"
                if kp.exists():
                    k = np.load(kp)
                    k = np.array([k[0, 0], k[1, 1], k[0, 2], k[1, 2]])
                else:
                    k = np.asarray(intrinsics)
                    used_fallback_k = True
            elif name == "droid_w":
                depth, rgb, k = depths[j], rgbs[j], ks[j]
            else:
                image = cv2.imread(str(frames[idx]))
                if image is None:
                    raise ValueError(f"unreadable RGB frame {idx}")
                rgb, k = prepared_rgb(image, cam)
                depth, mask = depths[j], masks[j]
            valid = np.isfinite(depth) & (depth > 1e-5)
            if mask is not None:
                valid &= mask
            # One fixed cutoff for the sequence; never rescale individual frames.
            if far_depth is None and np.any(valid) and far_ratio > 0:
                far_depth = float(np.median(depth[valid])) * far_ratio
            points, colors = points_from_depth(depth, rgb, scene_poses[idx], k,
                                               model["scale"], max_points, mask, far_depth)
            # A multiview consistency mask can reject moving people. Keep all
            # finite, in-range depths in the current-frame animation; apply the
            # model mask only to the accumulated diagnostic map.
            dynamic_points, dynamic_colors = points_from_depth(
                depth, rgb, scene_poses[idx], k, model["scale"], max_points,
                far_depth=far_depth)
            if not len(dynamic_points):
                continue
            dynamic_valid = np.isfinite(depth) & (depth > 1e-5)
            filtered_depth = np.where(dynamic_valid & (depth <= far_depth if far_depth is not None else True), depth, 0)
            snapshots.append(dict(frame=idx, points=points, colors=colors, k=k,
                                  dynamic_points=dynamic_points, dynamic_colors=dynamic_colors,
                                  rgb=rgb, depth=filtered_depth.astype(np.float32)))
        return dict(snapshots=snapshots, poses=scene_poses, source=source,
                    calibration=calibration, fallback_intrinsics=used_fallback_k,
                    far_depth=far_depth, scale=model["scale"],
                    mask="saved valid_depth_masks" if name == "wildpose" else "finite positive depth",
                    available_frames=len(timestamps))
    finally:
        if archive is not None:
            archive.close()


def new_recording(path, app_id, blueprint):
    import rerun as rr
    rec = rr.RecordingStream(app_id, recording_id=uuid4())
    rec.save(path)
    # Send after attaching the file sink, and make the saved layout active.
    rec.send_blueprint(blueprint, make_active=True, make_default=True)
    return rec


def log_camera(rec, root, pose, k, size, color):
    import rerun as rr
    rec.log(f"{root}/camera", rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3]))
    rec.log(f"{root}/camera", rr.Pinhole(focal_length=k[:2], principal_point=k[2:],
            resolution=size, camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=0.5, color=color))


def log_scene(rec, root):
    import rerun as rr
    rec.log(root, rr.ViewCoordinates.RDF, static=True)


def log_dynamic_frame(rec, root, scene, snap):
    """Replace geometry on one temporal entity, never log it as static."""
    import rerun as rr
    rec.log(f"{root}/points", rr.Points3D(
        snap["dynamic_points"], colors=snap["dynamic_colors"], radii=rr.Radius.ui_points(2.0)))
    # Keep the visible surface centered from an external orbit viewpoint.
    # Tracking a pinhole would instead take over the input camera's exact pose.
    center = (np.median(snap["dynamic_points"], axis=0) if len(snap["dynamic_points"])
              else scene["poses"][snap["frame"], :3, 3])
    rec.log(f"{root}/scene_anchor", rr.Points3D([center], colors=[[0, 0, 0, 0]], radii=0.01))


def scene_view(name, root, scene, follow=False):
    import rerun.blueprint as rrb
    # Start near the first camera and look into the scene; distant bounds do not
    # determine the initial eye. This is a viewer transform, not a geometry edit.
    points = scene["snapshots"][0]["dynamic_points"]
    target = np.median(points, axis=0)
    origin = scene["poses"][0, :3, 3]
    distance = max(float(np.linalg.norm(target-origin)), 1.0)
    return rrb.Spatial3DView(name=name, origin=root, eye_controls=rrb.EyeControls3D(
        position=origin + np.array([0, -0.2*distance, -0.3*distance]),
        look_target=target, eye_up=[0, -1, 0],
        tracking_entity=f"{root}/scene_anchor" if follow else None))


def export_scenes(output, frames, scenes, labels, colors, failures, options):
    import rerun as rr
    import rerun.blueprint as rrb
    report = dict(options=options, failures=failures, models={}, files=[])
    if not scenes:
        (output / "scene_diagnostics.json").write_text(json.dumps(report, indent=2))
        raise RuntimeError("no model has usable scene depth; see scene_diagnostics.json")
    blueprint = rrb.Blueprint(rrb.Grid(
        rrb.Spatial2DView(name="Input RGB", origin="/input"),
        *[scene_view(f"{labels[n]} - dynamic ({len(s['snapshots'])} depth frames)",
                     f"/scenes/{n}", s, follow=True) for n, s in scenes.items()],
        grid_columns=2), rrb.TimePanel(state="collapsed", timeline="frame",
            fps=options["fps"], play_state="playing", loop_mode="all"), collapse_panels=True)
    comparison_path = output / "scene_comparison.rrd"
    comparison = new_recording(comparison_path, "dynamic_ba_scenes_v3", blueprint)
    report["files"].append(str(comparison_path))
    try:
        for name, scene in scenes.items():
            log_scene(comparison, f"/scenes/{name}")
        by_frame = {n: {s["frame"]: s for s in scene["snapshots"]} for n, scene in scenes.items()}
        for idx, path in enumerate(frames):
            comparison.set_time("frame", sequence=idx)
            comparison.log("/input/rgb", rr.Image(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)).compress(jpeg_quality=90))
            for name, scene in scenes.items():
                snap = by_frame[name].get(idx)
                if snap is not None:
                    log_dynamic_frame(comparison, f"/scenes/{name}", scene, snap)
        comparison.flush()
    finally:
        comparison.disconnect()
    (output / "scenes").mkdir(exist_ok=True)
    for name, scene in scenes.items():
        blueprint = rrb.Blueprint(rrb.Grid(
            scene_view("Dynamic scene - current depth frame", "/world", scene, follow=True),
            scene_view("Accumulated scene - up to current frame", "/accumulated", scene),
            rrb.Spatial2DView(name="RGB - sampled depth frame", origin="/input"),
            rrb.Spatial2DView(name="Depth - same sampled frame", origin="/depth"),
            grid_columns=2), rrb.TimePanel(state="collapsed", timeline="frame",
                fps=options["fps"], play_state="playing", loop_mode="all"), collapse_panels=True)
        path = output / "scenes" / f"{name}.rrd"
        rec = new_recording(path, f"dynamic_ba_scene_{name}_v3", blueprint)
        try:
            for root in ("/world", "/accumulated"):
                log_scene(rec, root)
            for snap in scene["snapshots"]:
                idx = snap["frame"]
                rec.set_time("frame", sequence=idx)
                log_dynamic_frame(rec, "/world", scene, snap)
                # Unique entity per frame persists after its first timestamp. Going
                # backwards hides future entities; logging one path would replace it.
                rec.log(f"/accumulated/frames/{idx:06d}", rr.Points3D(
                    snap["points"], colors=snap["colors"], radii=rr.Radius.ui_points(1.2)))
                rec.log("/input/rgb", rr.Image(snap["rgb"]).compress(jpeg_quality=90))
                rec.log("/depth/image", rr.DepthImage(snap["depth"] * scene["scale"]))
            rec.flush()
        finally:
            rec.disconnect()
        stats = {k: scene[k] for k in ("source", "calibration", "fallback_intrinsics", "far_depth", "scale", "mask", "available_frames")}
        stats.update(sampled_frames=[s["frame"] for s in scene["snapshots"]],
                     point_count=sum(len(s["points"]) for s in scene["snapshots"]),
                     dynamic_point_count=sum(len(s["dynamic_points"]) for s in scene["snapshots"]),
                     dynamic_mask="finite positive depth plus fixed far cutoff; no multiview mask",
                     animation="current depth frame replaces previous geometry; hold last available depth between keyframes")
        report["models"][name] = stats
        report["files"].append(str(path))
        print(f"Scene {name}: {len(scene['snapshots'])} frames, {stats['point_count']:,} points -> {path}")
    (output / "scene_diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
