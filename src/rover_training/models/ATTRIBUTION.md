# Rock assets

Retrieved 2026-10-07.

NASA Astromaterials 3D (https://ares.jsc.nasa.gov/astromaterials3d/) answered
HTTP 200, but the public catalog page is a viewer. It did not link a direct
OBJ, DAE, or GLB download, so no NASA mesh is in this folder.

The meshes below are Gazebo Fuel models published by Open Robotics. License:
Creative Commons Attribution 4.0 International
(https://creativecommons.org/licenses/by/4.0/). Authors recorded in the
upstream `model.config`: Addisu Taddese (addisu@openrobotics.org) and Cole
Biesemeyer (cole@openrobotics.org).

| Folder | Fuel model | Visual mesh | Notes |
| --- | --- | --- | --- |
| `falling_rock1` | Falling Rock 1 | `meshes/FallingRock01.dae` | training asset |
| `falling_rock2` | Falling Rock 2 | `meshes/FallingRock02.dae` | training asset |
| `falling_rock3` | Falling Rock 3 | `meshes/FallingRock03.dae` | training asset |
| `falling_rock4` | Falling Rock 4 | `meshes/FallingRock04.dae` | held out for the test split |

Each DAE declares `unit meter="0.01"` and `up_axis` Z_UP. Gazebo Classic's
Collada loader applies that unit, so the measured size is about 0.45 m to
1.5 m before `nominal_scale` in `assets.json`. The training world scales each
rock so its longest side is about 0.30 m, then applies the random scale factor.

The Fuel collider mesh (`*_Collider.dae`) was not downloaded. Worlds use a
box around the visual mesh bounds. The albedo texture
`materials/textures/FallingRocks_Albedo.jpg` is the upstream diffuse map.
Normal and roughness maps were left out because Gazebo Classic uses the OGRE
script, which samples the albedo only.

`falling_rock4` is never placed in a train or val scene. Random scenes are
either training assets or the held-out asset, not a mixture.
