"""Rock asset catalog.

Meshes are Gazebo Fuel models from OpenRobotics, license CC-BY-4.0.
Bounds are meters after the Collada unit (meter="0.01") that Gazebo Classic
applies when it loads the DAE. Collision in the training world is the AABB,
not the upstream collider mesh.
"""

import json
import re
from pathlib import Path

CATALOG = (
    {
        "id": "falling_rock1",
        "fuel_name": "Falling Rock 1",
        "fuel_owner": "OpenRobotics",
        "fuel_url": "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Falling%20Rock%201",
        "mesh": "meshes/FallingRock01.dae",
        "material_script": "FallingRocks/FallingRocks_Diffuse",
        "held_out": False,
        "temporary": False,
    },
    {
        "id": "falling_rock2",
        "fuel_name": "Falling Rock 2",
        "fuel_owner": "OpenRobotics",
        "fuel_url": "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Falling%20Rock%202",
        "mesh": "meshes/FallingRock02.dae",
        "material_script": "FallingRocks/FallingRocks_Diffuse",
        "held_out": False,
        "temporary": False,
    },
    {
        "id": "falling_rock3",
        "fuel_name": "Falling Rock 3",
        "fuel_owner": "OpenRobotics",
        "fuel_url": "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Falling%20Rock%203",
        "mesh": "meshes/FallingRock03.dae",
        "material_script": "FallingRocks/FallingRocks_Diffuse",
        "held_out": False,
        "temporary": False,
    },
    {
        "id": "falling_rock4",
        "fuel_name": "Falling Rock 4",
        "fuel_owner": "OpenRobotics",
        "fuel_url": "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Falling%20Rock%204",
        "mesh": "meshes/FallingRock04.dae",
        "material_script": "FallingRocks/FallingRocks_Diffuse",
        "held_out": True,
        "temporary": False,
    },
)

TARGET_LONGEST_M = 0.30
LICENSE_NAME = "Creative Commons Attribution 4.0 International"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"


def models_dir():
    source = Path(__file__).resolve().parents[1] / "models"
    if (source / "assets.json").is_file():
        return source
    try:
        from ament_index_python.packages import get_package_share_directory
        shared = Path(get_package_share_directory("rover_training")) / "models"
        if (shared / "assets.json").is_file():
            return shared
    except Exception:
        pass
    return source


def catalog_path():
    return models_dir() / "assets.json"


def measure_dae(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    unit = re.search(r'<unit\b[^>]*meter="([^"]+)"', text)
    meters_per_unit = float(unit.group(1)) if unit else 1.0
    up = re.search(r"<up_axis>\s*([^<]+)\s*</up_axis>", text)
    up_axis = up.group(1).strip() if up else "Y_UP"
    chosen = None
    for attrs, body in re.findall(r"<float_array\b([^>]*)>([^<]+)</float_array>", text):
        ident = ""
        match = re.search(r'id="([^"]+)"', attrs)
        if match:
            ident = match.group(1)
        if "position" not in ident.lower():
            continue
        values = [float(item) for item in body.split()]
        if len(values) < 9 or len(values) % 3:
            continue
        if chosen is None or len(values) > len(chosen[1]):
            chosen = (ident, values)
    if chosen is None:
        raise ValueError(f"no position array in {path}")
    points = list(zip(chosen[1][0::3], chosen[1][1::3], chosen[1][2::3]))
    if up_axis == "Y_UP":
        points = [(x, -z, y) for x, y, z in points]
    elif up_axis == "X_UP":
        points = [(-y, z, x) for x, y, z in points]
    elif up_axis != "Z_UP":
        raise ValueError(f"unsupported up_axis {up_axis}")
    xs, ys, zs = zip(*points)
    bounds_min = [min(xs) * meters_per_unit, min(ys) * meters_per_unit, min(zs) * meters_per_unit]
    bounds_max = [max(xs) * meters_per_unit, max(ys) * meters_per_unit, max(zs) * meters_per_unit]
    longest = max(bounds_max[i] - bounds_min[i] for i in range(3))
    return {
        "position_id": chosen[0],
        "vertices": len(points),
        "up_axis": up_axis,
        "file_unit_meters": meters_per_unit,
        "bounds_min_m": bounds_min,
        "bounds_max_m": bounds_max,
        "nominal_scale": TARGET_LONGEST_M / longest,
    }


def build_catalog(root=None):
    root = Path(root) if root else models_dir()
    assets = []
    for record in CATALOG:
        measured = measure_dae(root / record["id"] / record["mesh"])
        asset = dict(record)
        asset.update(measured)
        asset["class_name"] = "rock"
        asset["license"] = LICENSE_NAME
        asset["license_url"] = LICENSE_URL
        assets.append(asset)
    return {
        "class_names": ["rock"],
        "collision": "axis-aligned box in the mesh frame, not the Fuel collider mesh",
        "nasa_astromaterials_3d": {
            "url": "https://ares.jsc.nasa.gov/astromaterials3d/",
            "retrieved": "2026-10-07",
            "result": "The public catalog page is an interactive viewer and did not provide a direct OBJ or GLB URL.",
        },
        "assets": assets,
    }


def write_catalog(path=None, root=None):
    path = Path(path) if path else catalog_path()
    document = build_catalog(root)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document


def write_model_files(root=None):
    """Write assets.json plus a Classic model.config that uses box collision."""
    root = Path(root) if root else models_dir()
    document = write_catalog(root / "assets.json", root)
    for asset in document["assets"]:
        folder = root / asset["id"]
        upstream = folder / "model.config.upstream"
        if upstream.exists():
            upstream.unlink()
        (folder / "model.config").write_text(_model_config(asset), encoding="utf-8")
        (folder / "model.sdf").write_text(_model_sdf(asset), encoding="utf-8")
    return document


def _model_config(asset):
    return f"""<?xml version="1.0"?>
<model>
  <name>{asset['id']}</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <author>
    <name>Addisu Taddese and Cole Biesemeyer, Open Robotics</name>
    <email>addisu@openrobotics.org</email>
  </author>
  <description>
    Gazebo Fuel model "{asset['fuel_name']}" by {asset['fuel_owner']}.
    {asset['license']}. The visual mesh is upstream. This SDF replaces the
    collider mesh with a box. Source: {asset['fuel_url']}
  </description>
</model>
"""


def _model_sdf(asset):
    scale = float(asset["nominal_scale"])
    low = asset["bounds_min_m"]
    high = asset["bounds_max_m"]
    size = tuple((high[i] - low[i]) * scale for i in range(3))
    center = tuple(((high[i] + low[i]) * 0.5) * scale for i in range(3))
    z = -low[2] * scale
    mesh = asset["mesh"]
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{asset['id']}">
    <static>true</static>
    <pose>0 0 {z:.4f} 0 0 0</pose>
    <link name="link">
      <collision name="box">
        <pose>{center[0]:.4f} {center[1]:.4f} {center[2]:.4f} 0 0 0</pose>
        <geometry><box><size>{size[0]:.4f} {size[1]:.4f} {size[2]:.4f}</size></box></geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <mesh>
            <uri>{mesh}</uri>
            <scale>{scale:.6f} {scale:.6f} {scale:.6f}</scale>
          </mesh>
        </geometry>
        <material>
          <ambient>0.35 0.32 0.28 1</ambient>
          <diffuse>0.45 0.40 0.34 1</diffuse>
          <script>
            <uri>model://{asset['id']}/materials/scripts</uri>
            <uri>model://{asset['id']}/materials/textures</uri>
            <name>{asset['material_script']}</name>
          </script>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


def load_assets(path=None):
    path = Path(path) if path else catalog_path()
    document = json.loads(path.read_text(encoding="utf-8"))
    return document["assets"]


def asset_by_id(assets, asset_id):
    for asset in assets:
        if asset["id"] == asset_id:
            return asset
    raise KeyError(asset_id)
