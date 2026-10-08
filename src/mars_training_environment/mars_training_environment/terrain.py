"""A sampled height field shared by the world, placement and annotation."""
import math
from pathlib import Path

import numpy as np

from .common import png, write_json
from .meshes import write_dae


class Terrain:
    def __init__(self, heights, size):
        self.h = np.asarray(heights, dtype=float)
        self.size = float(size)
        self.n = len(self.h)
        self.step = self.size / (self.n - 1)

    def height(self, x, y):
        """Interpolate the same diagonal as the explicit collision triangles."""
        u = np.clip((np.asarray(x)+self.size/2)/self.step, 0, self.n-1-1e-9)
        v = np.clip((np.asarray(y)+self.size/2)/self.step, 0, self.n-1-1e-9)
        i, j = u.astype(int), v.astype(int)
        a, b = u-i, v-j
        z00, z10 = self.h[j, i], self.h[j, i+1]
        z01, z11 = self.h[j+1, i], self.h[j+1, i+1]
        return np.where(a+b <= 1, z00 + a*(z10-z00)+b*(z01-z00),
                        z11+(1-a)*(z01-z11)+(1-b)*(z10-z11))

    def gradient(self, x, y):
        d = self.step * .2
        return np.array([(self.height(x+d,y)-self.height(x-d,y))/(2*d),
                         (self.height(x,y+d)-self.height(x,y-d))/(2*d)])

    def mesh(self):
        coords = np.linspace(-self.size/2, self.size/2, self.n)
        x, y = np.meshgrid(coords, coords)
        vertices = np.column_stack([x.ravel(), y.ravel(), self.h.ravel()])
        faces = []
        for j in range(self.n-1):
            for i in range(self.n-1):
                a = j*self.n+i
                faces.extend([[a, a+1, a+self.n], [a+1, a+self.n+1, a+self.n]])
        return vertices, np.asarray(faces, dtype=int)

    def maximum_slope(self):
        v, f = self.mesh()
        normals = np.cross(v[f[:,1]]-v[f[:,0]], v[f[:,2]]-v[f[:,0]])
        return float(np.max(np.arctan2(np.linalg.norm(normals[:, :2], axis=1), normals[:, 2])))


def generate_terrain(seed, config, folder):
    rng = np.random.RandomState(seed)
    n, size = config['grid_size'], config['size_m']
    coords = np.linspace(-size/2, size/2, n)
    x, y = np.meshgrid(coords, coords)
    h = np.zeros_like(x)
    # Broad undulations, elongated ridges and shallow negative Gaussian bowls.
    for _ in range(9):
        angle, phase = rng.uniform(0, 2*math.pi, 2)
        frequency = rng.uniform(.2, 1.5)
        h += rng.uniform(.03, .14)*np.sin((x*math.cos(angle)+y*math.sin(angle))*frequency+phase)
    for _ in range(7):
        cx, cy = rng.uniform(-size*.4, size*.4, 2)
        sx, sy = rng.uniform(.9, 2.8, 2)
        h += rng.uniform(-.4, .4)*np.exp(-((x-cx)/sx)**2-((y-cy)/sy)**2)
    # C1 smooth flat pad centered on the unchanged rover spawn location.
    distance = np.hypot(x, y)
    blend = np.clip((distance-config['spawn_clearance_m'])/1.2, 0, 1)
    blend = blend*blend*(3-2*blend)
    h = (h-h[n//2,n//2])*blend
    h *= config['relief_m']/max(float(np.ptp(h)), 1e-8)
    field = Terrain(h, size)
    slope = field.maximum_slope()
    h *= min(1, math.tan(math.radians(config['max_slope_deg']))/max(math.tan(slope), 1e-8)) * .93
    # Classic supports grayscale heightmaps. Placement uses the quantized heights,
    # never an unquantized approximation to the rendered/collision surface.
    low, high = float(h.min()), float(h.max())
    encoded = np.round((h-low)/(high-low)*255).astype(np.uint8)
    h = low + encoded.astype(float)/255*(high-low)
    # Make the flat pad exactly z=0; retain quantization spacing elsewhere.
    zero = float(h[n//2,n//2])
    h -= zero
    low -= zero
    field = Terrain(h, size)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    png(folder/'heightmap.png', np.flipud(encoded))
    texture_n = config['texture_size']
    yy, xx = np.mgrid[0:texture_n, 0:texture_n]/texture_n
    phase = rng.uniform(0, 6.28, 4)
    patches = (np.sin(xx*19+phase[0]) * np.cos(yy*15+phase[1]) + np.sin((xx+yy)*31+phase[2]))*.5
    dune = np.sin(xx*300+np.sin(yy*70)*2)*1.8
    noise = rng.normal(0, 5, (texture_n, texture_n))
    red = np.array([139, 83, 58])+rng.uniform(-8, 8, 3)
    tan = np.array([173, 143, 106])
    gray = np.array([111, 109, 105])
    texture = red + np.maximum(patches, 0)[:, :, None]*(tan-red) + np.maximum(-patches,0)[:, :, None]*(gray-red)
    texture += (noise+dune)[:, :, None]
    png(folder/'ground.png', np.clip(texture, 0, 255))
    png(folder/'normal.png', np.tile([128,128,255], (8,8,1)))
    v, f = field.mesh()
    write_dae(folder/'terrain.dae', v, f, texture='ground.png')
    document = {'heights': h.tolist(), 'size_m': size, 'heightmap_min_m': low,
                'heightmap_range_m': float(h.max()-h.min()), 'maximum_slope_deg': math.degrees(field.maximum_slope()),
                'backend': config['backend'], 'grid_size': n, 'spawn_z_m': float(field.height(0.,0.))}
    write_json(folder/'terrain.json', document)
    return field, document
