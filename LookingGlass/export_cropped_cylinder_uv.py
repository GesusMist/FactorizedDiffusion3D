"""Export a cylinder-only ray-traced UV map from the current Blender camera.

Run this inside Blender's Python editor after setting up:
  - active scene camera
  - image plane object named LG_Plane
  - mirror cylinder object named LG_Mirror

The script first scans the whole camera frame to find pixels that see the
cylindrical mirror and reflect onto the image plane. It then crops that valid
mirror region and resamples only the crop to a full UV map. This is the map the
LookingGlass notebook should use: the hidden-view image is the cylinder surface,
not the entire camera render containing the plane and background.
"""

import json
import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


PLANE_NAME = "LG_Plane"
MIRROR_NAME = "LG_Mirror"

# Use the existing active scene camera. The script never changes camera pose.
SCAN_RES_X = 512
SCAN_RES_Y = 512
OUT_RES_X = 1024
OUT_RES_Y = 1024

# Add a little context around the detected mirror surface.
CROP_PADDING_FRACTION = 0.04
CROP_PADDING_PIXELS = 8

MAX_DIST = 100.0
EPS = 1e-4

OUTPUT_DIR_NAME = "lookingglass_uv"
OUTPUT_NPY_NAME = "cylinder_raytraced_uv_cropped.npy"
OUTPUT_META_NAME = "cylinder_raytraced_uv_cropped_metadata.json"


scene = bpy.context.scene
depsgraph = bpy.context.evaluated_depsgraph_get()

camera = scene.camera
if camera is None:
    raise RuntimeError("No active scene camera found. Select a camera and use Ctrl+0 to make it active.")

plane = bpy.data.objects.get(PLANE_NAME)
mirror = bpy.data.objects.get(MIRROR_NAME)
if plane is None:
    raise RuntimeError(f"Missing plane object named {PLANE_NAME!r}.")
if mirror is None:
    raise RuntimeError(f"Missing mirror object named {MIRROR_NAME!r}.")


def camera_ray(cam, px, py, res_x, res_y):
    """Return a world-space ray through a pixel in the current camera frame."""
    frame = cam.data.view_frame(scene=scene)
    top_right = cam.matrix_world @ frame[0]
    bottom_right = cam.matrix_world @ frame[1]
    bottom_left = cam.matrix_world @ frame[2]
    top_left = cam.matrix_world @ frame[3]

    u = (px + 0.5) / res_x
    v = (py + 0.5) / res_y
    top = top_left.lerp(top_right, u)
    bottom = bottom_left.lerp(bottom_right, u)
    target = top.lerp(bottom, v)

    if cam.data.type == "ORTHO":
        direction = (cam.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))).normalized()
        origin = target
    else:
        origin = cam.matrix_world.translation
        direction = (target - origin).normalized()
    return origin, direction


def cylinder_normal_world(obj, world_loc):
    """Analytic radial normal for a vertical local-Z cylinder."""
    local_loc = obj.matrix_world.inverted() @ world_loc
    local_normal = Vector((local_loc.x, local_loc.y, 0.0))
    if local_normal.length == 0:
        return None
    normal_matrix = obj.matrix_world.to_3x3().inverted().transposed()
    return (normal_matrix @ local_normal).normalized()


def plane_hit_to_image_uv(obj, world_loc):
    """Map a hit on LG_Plane to image UV: u left->right, v top->bottom."""
    local = obj.matrix_world.inverted() @ world_loc
    verts = obj.data.vertices
    min_x = min(v.co.x for v in verts)
    max_x = max(v.co.x for v in verts)
    min_y = min(v.co.y for v in verts)
    max_y = max(v.co.y for v in verts)

    if math.isclose(max_x, min_x) or math.isclose(max_y, min_y):
        return None

    u = (local.x - min_x) / (max_x - min_x)
    v = 1.0 - (local.y - min_y) / (max_y - min_y)

    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return None
    return float(u), float(v)


def trace_uv_at_camera_pixel(px, py, res_x, res_y):
    """Trace camera -> mirror -> plane and return canonical image UV or None."""
    origin, direction = camera_ray(camera, px, py, res_x, res_y)
    hit, loc, _normal, _face_index, obj, _matrix = scene.ray_cast(
        depsgraph, origin, direction, distance=MAX_DIST
    )
    if not hit or obj is None or obj.name != MIRROR_NAME:
        return None

    normal = cylinder_normal_world(mirror, loc)
    if normal is None:
        return None

    reflected = (direction - 2.0 * direction.dot(normal) * normal).normalized()
    hit2, loc2, _normal2, _face_index2, obj2, _matrix2 = scene.ray_cast(
        depsgraph, loc + reflected * EPS, reflected, distance=MAX_DIST
    )
    if not hit2 or obj2 is None or obj2.name != PLANE_NAME:
        return None

    return plane_hit_to_image_uv(plane, loc2)


def find_crop_box():
    mask = np.zeros((SCAN_RES_Y, SCAN_RES_X), dtype=bool)
    for y in range(SCAN_RES_Y):
        if y % 64 == 0:
            print(f"scan row {y}/{SCAN_RES_Y}", flush=True)
        for x in range(SCAN_RES_X):
            mask[y, x] = trace_uv_at_camera_pixel(x, y, SCAN_RES_X, SCAN_RES_Y) is not None

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise RuntimeError(
            "No valid mirror pixels found. In camera view, LG_Mirror must be visible, "
            "and reflected rays from it must hit LG_Plane."
        )

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad = max(
        CROP_PADDING_PIXELS,
        int(round(max(x1 - x0, y1 - y0) * CROP_PADDING_FRACTION)),
    )
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(SCAN_RES_X, x1 + pad)
    y1 = min(SCAN_RES_Y, y1 + pad)
    return x0, y0, x1, y1, int(mask.sum())


def export_cropped_uv(crop_box):
    x0, y0, x1, y1 = crop_box
    uv = np.full((OUT_RES_Y, OUT_RES_X, 2), np.nan, dtype=np.float32)

    for y in range(OUT_RES_Y):
        if y % 64 == 0:
            print(f"export row {y}/{OUT_RES_Y}", flush=True)
        src_y = y0 + (y + 0.5) * (y1 - y0) / OUT_RES_Y - 0.5
        for x in range(OUT_RES_X):
            src_x = x0 + (x + 0.5) * (x1 - x0) / OUT_RES_X - 0.5
            coord = trace_uv_at_camera_pixel(src_x, src_y, SCAN_RES_X, SCAN_RES_Y)
            if coord is not None:
                uv[y, x, 0] = coord[0]
                uv[y, x, 1] = coord[1]
    return uv


blend_path = Path(bpy.data.filepath).resolve() if bpy.data.filepath else Path.cwd()
base_dir = blend_path.parent if blend_path.suffix else blend_path
out_dir = base_dir / OUTPUT_DIR_NAME
out_dir.mkdir(parents=True, exist_ok=True)

print("Finding cylinder crop in current camera frame...", flush=True)
crop_x0, crop_y0, crop_x1, crop_y1, scan_valid = find_crop_box()
print(
    f"crop box in {SCAN_RES_X}x{SCAN_RES_Y} camera scan: "
    f"x={crop_x0}:{crop_x1}, y={crop_y0}:{crop_y1}; valid scan pixels={scan_valid}",
    flush=True,
)

uv_map = export_cropped_uv((crop_x0, crop_y0, crop_x1, crop_y1))
valid = np.isfinite(uv_map).all(axis=-1)

out_npy = out_dir / OUTPUT_NPY_NAME
np.save(out_npy, uv_map)

metadata = {
    "camera": camera.name,
    "plane": PLANE_NAME,
    "mirror": MIRROR_NAME,
    "scan_resolution": [SCAN_RES_X, SCAN_RES_Y],
    "output_resolution": [OUT_RES_X, OUT_RES_Y],
    "crop_box_xyxy_in_scan_pixels": [crop_x0, crop_y0, crop_x1, crop_y1],
    "scan_valid_pixels": scan_valid,
    "output_valid_pixels": int(valid.sum()),
    "output_valid_fraction": float(valid.mean()),
    "output_npy": str(out_npy),
}
(out_dir / OUTPUT_META_NAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print(f"saved: {out_npy}", flush=True)
print(f"valid output pixels: {valid.sum():,} / {valid.size:,} ({valid.mean():.2%})", flush=True)
print(f"metadata: {out_dir / OUTPUT_META_NAME}", flush=True)
