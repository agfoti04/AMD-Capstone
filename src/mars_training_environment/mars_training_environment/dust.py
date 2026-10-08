"""Coat a scanned rock with the dust colour of the scene it sits in.

A multiply tint cannot lift a near-black lunar albedo into a dusty tan, so the
albedo and roughness are rewritten. Upward faces take more dust than
undersides. The result is a texture; Gazebo's boxes and our labels are unchanged.
"""
import json
from pathlib import Path

import numpy as np


def dust_images(albedo, roughness, vertices, uvs, faces, dust_bgr):
    """Return albedo and roughness arrays. `dust_bgr` is 0..255."""
    import cv2
    albedo = np.asarray(albedo, dtype=np.float32)
    roughness = np.asarray(roughness, dtype=np.float32)
    if roughness.ndim == 3:
        roughness = roughness[:, :, 0]
    height, width = albedo.shape[:2]
    vertices = np.asarray(vertices, dtype=float)
    uvs = np.asarray(uvs, dtype=float)
    faces = np.asarray(faces, dtype=int)
    triangles = vertices[faces]
    normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    up = normal[:, 2]/np.maximum(np.linalg.norm(normal, axis=1), 1e-9)
    # Tops take a heavy coat. Sides keep most of the scanned colour, so a
    # textured rock does not become a flat red shape. The blur removes the
    # hard edge between neighbouring faces.
    cover = 0.12+0.78*np.clip(up, 0, 1)
    painted = np.zeros((height, width), dtype=np.uint8)
    for index in np.argsort(cover):
        points = np.column_stack([uvs[faces[index], 0]*width, (1-uvs[faces[index], 1])*height])
        cv2.fillConvexPoly(painted, np.round(points).astype(np.int32), int(round(cover[index]*255)))
    blur = max(2.0, min(height, width)/90)
    factor = cv2.GaussianBlur(painted, (0, 0), blur).astype(np.float32)/255
    factor = factor[:, :, None]
    dust = np.asarray(dust_bgr, dtype=np.float32).reshape(1, 1, 3)
    coated = albedo*(1-factor)+dust*factor
    coated_rough = roughness*(1-factor[:, :, 0])+235*factor[:, :, 0]
    return np.clip(coated, 0, 255).astype(np.uint8), np.clip(coated_rough, 0, 255).astype(np.uint8)


def bake_scene_dust(bundle, scene_folder, assets, objects, dust_bgr):
    """Write one dusty texture set per scanned asset used in the scene.

    Paths are relative to the bundle. Rocks without UVs keep their own maps.
    """
    import cv2
    bundle = Path(bundle)
    used = {obj['asset_id'] for obj in objects}
    baked = {}
    quality = [cv2.IMWRITE_JPEG_QUALITY, 92]
    for asset in assets:
        if asset['id'] not in used:
            continue
        geometry = json.loads((bundle/asset['geometry']).read_text())
        if asset.get('shared_shape') or 'uv' not in geometry or 'roughness' not in asset['textures']:
            continue
        albedo = cv2.imread(str(bundle/asset['textures']['albedo']), cv2.IMREAD_COLOR)
        roughness = cv2.imread(str(bundle/asset['textures']['roughness']), cv2.IMREAD_GRAYSCALE)
        if albedo is None or roughness is None:
            raise FileNotFoundError(f'Missing texture for {asset["id"]}')
        coated, coated_rough = dust_images(albedo, roughness, geometry['vertices'], geometry['uv'],
                                            geometry['faces'], dust_bgr)
        dest = Path(scene_folder)/'dust'/asset['id']
        dest.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest/'albedo.jpg'), coated, quality)
        cv2.imwrite(str(dest/'roughness.jpg'), coated_rough, quality)
        normal = bundle/asset['textures']['normal']
        baked[asset['id']] = {
            'albedo': dest.relative_to(bundle).as_posix()+'/albedo.jpg',
            'normal': asset['textures']['normal'],
            'roughness': dest.relative_to(bundle).as_posix()+'/roughness.jpg',
        }
        if not normal.is_file():
            raise FileNotFoundError(f'Missing normal map for {asset["id"]}')
    return baked
