"""SDF 1.6 worlds for Gazebo Classic. This is not a Gazebo Harmonic world."""

import math

ROBOT_SPAWN_Z = 0.09


def sun_direction(azimuth, elevation):
    horizontal = math.cos(elevation)
    return (
        horizontal * math.cos(azimuth),
        horizontal * math.sin(azimuth),
        -math.sin(elevation),
    )


def build_world_sdf(scene, assets):
    by_id = {asset["id"]: asset for asset in assets}
    rocks = []
    for rock in scene["rocks"]:
        asset = by_id[rock["asset_id"]]
        rocks.append(_rock_model(rock, asset))
    direction = sun_direction(scene["sun_azimuth"], scene["sun_elevation"])
    ambient = float(scene["ambient"])
    brightness = float(scene["sun_brightness"])
    body = "\n".join(rocks)
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <world name="rover_rocks">
    <gravity>0 0 -9.8</gravity>
    <scene>
      <ambient>{ambient:.3f} {ambient:.3f} {ambient:.3f} 1</ambient>
      <background>0.55 0.62 0.70 1</background>
      <shadows>true</shadows>
    </scene>
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>{brightness:.3f} {brightness:.3f} {brightness:.3f} 1</diffuse>
      <specular>0.15 0.15 0.15 1</specular>
      <direction>{direction[0]:.4f} {direction[1]:.4f} {direction[2]:.4f}</direction>
    </light>
    <include><uri>model://ground_plane</uri></include>
    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
      <ros><namespace>/gazebo</namespace></ros>
      <update_rate>30</update_rate>
    </plugin>
{body}
  </world>
</sdf>
"""


def write_world(path, scene, assets):
    text = build_world_sdf(scene, assets)
    path.write_text(text, encoding="utf-8")
    return path


def _rock_model(rock, asset):
    scale = float(rock["scale"])
    pose = rock["xyz"]
    rpy = rock["rpy"]
    low = asset["bounds_min_m"]
    high = asset["bounds_max_m"]
    size = tuple((high[i] - low[i]) * scale for i in range(3))
    center = tuple((high[i] + low[i]) * 0.5 * scale for i in range(3))
    model = f"model://{asset['id']}"
    mesh = f"{model}/{asset['mesh']}"
    return f"""    <model name="{rock['name']}">
      <static>true</static>
      <pose>{pose[0]:.4f} {pose[1]:.4f} {pose[2]:.4f} {rpy[0]:.4f} {rpy[1]:.4f} {rpy[2]:.4f}</pose>
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
              <uri>{model}/materials/scripts</uri>
              <uri>{model}/materials/textures</uri>
              <name>{asset['material_script']}</name>
            </script>
          </material>
        </visual>
      </link>
    </model>"""
