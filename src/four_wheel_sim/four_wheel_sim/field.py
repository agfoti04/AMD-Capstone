# Open field: x from -1 to 9, y from -4 to 4. Robot starts at (0, 0).
# WALLS are (center_x, center_y, size_x, size_y) like the maze.
WALLS = [
    (4.0,  4.0, 10.0, 0.1),   # top
    (4.0, -4.0, 10.0, 0.1),   # bottom
    (-1.0, 0.0, 0.1, 8.0),    # left
    (9.0,  0.0, 0.1, 8.0),    # right
]

# OBSTACLES are (kind, x, y, a, b, material)
#   box:      a = size_x, b = size_y
#   cylinder: a = radius, b unused
OBSTACLES = [
    ('box',      4.0,  0.0, 0.4, 3.0, 'Gazebo/Wood'),    # barrier on the direct line
    ('cylinder', 2.0, -1.5, 0.4, 0.0, 'Gazebo/Red'),
    ('box',      6.0,  1.8, 1.2, 1.2, 'Gazebo/Blue'),
    ('cylinder', 6.5, -1.2, 0.5, 0.0, 'Gazebo/Green'),
    ('box',      7.0,  0.5, 0.5, 0.5, 'Gazebo/Orange'),
]

GOAL = (8.0, 0.0)