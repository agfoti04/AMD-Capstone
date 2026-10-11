import math

# Field: x from -1 to 13, y from -5 to 5. Robot starts at (0, 0) facing +x.
WALLS = [
    (6.0,  5.0, 14.0, 0.1),   # top
    (6.0, -5.0, 14.0, 0.1),   # bottom
    (-1.0, 0.0, 0.1, 10.0),   # left
    (13.0, 0.0, 0.1, 10.0),   # right
]

# Designated scouting region (x_min, x_max, y_min, y_max)
REGION = (1.0, 12.0, -4.0, 4.0)

# Rocks that sit on the lanes, so the route has to detour: (kind, x, y, a, b, material)
OBSTACLES = [
    ('cylinder', 5.5, -3.0, 0.4, 0.0, 'Gazebo/Grey'),
    ('box',      8.0,  1.0, 0.8, 0.8, 'Gazebo/Wood'),
    ('box',      4.0,  3.0, 0.7, 1.2, 'Gazebo/Bricks'),
]

# Sample targets (ground truth for scoring detections): (kind, x, y, size, material)
TARGETS = [
    ('box',     3.5, -2.0, 0.30, 'Gazebo/Red'),
    ('sphere',  5.0,  0.5, 0.35, 'Gazebo/Green'),
    ('sphere',  7.0, -3.6, 0.35, 'Gazebo/Blue'),
    ('box',     8.6,  1.0, 0.30, 'Gazebo/Turquoise'),  # hidden behind the wooden box
    ('box',     9.0,  2.2, 0.30, 'Gazebo/Purple'),
    ('box',    10.0, -1.0, 0.30, 'Gazebo/Yellow'),
    ('sphere', 11.0,  3.6, 0.35, 'Gazebo/Orange'),
]

# Tall posts for SLAM features: (x, y, material)
LANDMARKS = [
    (-0.5, -4.5, 'Gazebo/Gold'), (-0.5, 4.5, 'Gazebo/Bricks'),
    (6.0, -4.5, 'Gazebo/Wood'),  (6.0, 4.5, 'Gazebo/Gold'),
    (12.5, -4.5, 'Gazebo/Bricks'), (12.5, 4.5, 'Gazebo/Wood'),
    (6.5, -1.0, 'Gazebo/Gold'),
]

# Visual-only floor patches, no collision: (x, y, size_x, size_y, material)
PATCHES = [
    (3.0, -2.0, 3.0, 3.0, 'Gazebo/Grass'),
    (8.0,  2.0, 4.0, 3.0, 'Gazebo/WoodFloor'),
    (10.0, -3.0, 3.0, 2.0, 'Gazebo/Wood'),
    (5.0,  3.0, 2.0, 2.0, 'Gazebo/Gold'),
    (6.0,  0.0, 2.0, 2.0, 'Gazebo/Bricks'),
]

GOAL = (12.0, 0.0)


def lawnmower(region, spacing=2.0, margin=1.0):
    """Boustrophedon waypoints as (x, y, yaw). Lanes run along x."""
    x0, x1, y0, y1 = region
    pts, forward, y = [], True, y0 + margin
    while y <= y1 - margin + 1e-6:
        xs = (x0 + margin, x1 - margin)
        if not forward:
            xs = xs[::-1]
        yaw = 0.0 if forward else math.pi
        pts += [(x, y, yaw) for x in xs]
        forward, y = not forward, y + spacing
    return pts


LAWNMOWER = lawnmower(REGION)   # 4 lanes at y = -3, -1, 1, 3