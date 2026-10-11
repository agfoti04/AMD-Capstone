import os
from . import maze, field, survey

_WORLDS = {'maze': maze, 'field': field, 'survey': survey}
_w = _WORLDS[os.environ.get('SIM_WORLD', 'survey')]

WALLS = _w.WALLS
OBSTACLES = getattr(_w, 'OBSTACLES', [])
TARGETS = getattr(_w, 'TARGETS', [])
LANDMARKS = getattr(_w, 'LANDMARKS', [])
PATCHES = getattr(_w, 'PATCHES', [])
LAWNMOWER = getattr(_w, 'LAWNMOWER', [])
GOAL = _w.GOAL