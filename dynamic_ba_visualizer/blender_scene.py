"""Build a reproducible Blender trajectory animation from scene_data.json.

Usage:
  blender --background --python blender_scene.py -- DATA_JSON OUT_BLEND FRAME_DIR FPS
"""

import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector


def material(name, color, emission=0.0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1.0)
    if emission:
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        emission_input = bsdf.inputs.get("Emission Color") or bsdf.inputs.get("Emission")
        emission_input.default_value = (*color, 1.0)
        bsdf.inputs["Emission Strength"].default_value = emission
    return mat


def curve_object(name, points, color):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"; curve.bevel_depth = 0.025; curve.bevel_resolution = 3
    spline = curve.splines.new("POLY"); spline.points.add(len(points)-1)
    for dst, xyz in zip(spline.points, points):
        dst.co = (*xyz, 1.0)
    obj = bpy.data.objects.new(name, curve)
    obj.data.materials.append(material(name+"_mat", color, 0.25))
    bpy.context.collection.objects.link(obj)
    return obj


def add_label(text, location, color):
    curve = bpy.data.curves.new(text+"_font", "FONT"); curve.body = text
    curve.align_x = "CENTER"; curve.size = 0.35; curve.extrude = 0.01
    obj = bpy.data.objects.new(text, curve); obj.location = location
    obj.rotation_euler = (math.radians(75), 0, 0)
    obj.data.materials.append(material(text+"_mat", color, 0.2))
    bpy.context.collection.objects.link(obj)


args = sys.argv[sys.argv.index("--")+1:]
data_path, blend_path, frame_dir, fps = Path(args[0]), Path(args[1]), Path(args[2]), float(args[3])
data = json.loads(data_path.read_text())
bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete(use_global=False)

models = list(data["models"].items())
all_points = []
for model_index, (name, item) in enumerate(models):
    points = [Vector(p) for p in item["positions"]]
    offset = Vector(((model_index-(len(models)-1)/2)*8.0, 0, 0))
    points = [p+offset for p in points]
    all_points.extend(points)
    color = item["color"]
    curve_object(name+"_trajectory", points, color)
    add_label(item["label"], points[0]+Vector((0,0,1.0)), color)
    bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=.28, radius2=.05, depth=.6, location=points[0])
    marker = bpy.context.object; marker.name = name+"_current_camera"
    marker.rotation_mode = "QUATERNION"
    marker.data.materials.append(material(name+"_camera_mat", color, 0.4))
    for frame in range(data["frame_count"]):
        i = min(frame, len(points)-1); marker.location = points[i]
        marker.rotation_quaternion = Matrix(item["rotations"][i]).to_quaternion()
        marker.keyframe_insert("location", frame=frame+1)
        marker.keyframe_insert("rotation_quaternion", frame=frame+1)

center = sum(all_points, Vector()) / max(1, len(all_points))
extent = max((p-center).length for p in all_points) if all_points else 5
bpy.ops.mesh.primitive_plane_add(size=max(20, extent*3), location=(center.x, center.y, center.z-1.2))
plane = bpy.context.object; plane.data.materials.append(material("ground", (.035,.045,.06)))

bpy.ops.object.light_add(type="AREA", location=(center.x, center.y-4, center.z+10))
bpy.context.object.data.energy = 1800; bpy.context.object.data.shape = "DISK"; bpy.context.object.data.size = 8
bpy.ops.object.camera_add(location=(center.x, center.y-extent*1.6-8, center.z+extent*1.25+6))
camera = bpy.context.object; bpy.context.scene.camera = camera
direction = center - camera.location; camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

scene = bpy.context.scene
try:
    scene.render.engine = "BLENDER_EEVEE_NEXT"
except TypeError:
    scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = 1600; scene.render.resolution_y = 900; scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"; scene.render.fps = round(fps)
scene.frame_start = 1; scene.frame_end = data["frame_count"]
scene.world.color = (.008,.01,.018)
frame_dir.mkdir(parents=True, exist_ok=True); blend_path.parent.mkdir(parents=True, exist_ok=True)
scene.render.filepath = str(frame_dir / "frame_")
bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
bpy.ops.render.render(animation=True)
