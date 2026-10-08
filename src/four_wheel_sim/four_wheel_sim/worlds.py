import os
from . import maze, field

_w = field if os.environ.get('SIM_WORLD', 'field') == 'field' else maze

WALLS = _w.WALLS
OBSTACLES = getattr(_w, 'OBSTACLES', [])
GOAL = _w.GOAL