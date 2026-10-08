"""Per-scene ground material: a seamless blend of scanned texture sets."""
import math
from pathlib import Path

import numpy as np

MARS_RGB = np.array([0.78, 0.47, 0.31])


def periodic_field(rng, size, terms=5):
    """Smooth noise that repeats exactly every `size` pixels in x and y."""
    axis = np.arange(size, dtype=np.float32)*np.float32(2*math.pi/size)
    field = np.zeros((size, size), dtype=np.float32)
    for _ in range(terms):
        fx, fy = rng.randint(-4, 5, 2)
        if fx == 0 and fy == 0:
            fx = 1
        phase = np.float32(rng.uniform(0, 2*math.pi))
        amplitude = np.float32(rng.uniform(.5, 1)/math.hypot(fx, fy))
        field += amplitude*np.sin(np.add.outer(fy*axis+phase, fx*axis))
    return field


def broad_field(rng, size, terms=5, max_cycles=3.0):
    """Smooth variation across one image. Edges are not required to match."""
    axis = np.linspace(0, 1, size, dtype=np.float32)
    field = np.zeros((size, size), dtype=np.float32)
    for _ in range(terms):
        fx, fy = rng.uniform(0.35, max_cycles, 2)
        phase = np.float32(rng.uniform(0, 2*math.pi))
        amplitude = np.float32(rng.uniform(.5, 1)/math.hypot(fx, fy))
        field += amplitude*np.sin(np.float32(2*math.pi)*(fy*axis[:, None]+fx*axis[None, :])+phase)
    return field


def blend_weights(rng, layers, size, sharpness=6.0):
    fields = np.stack([periodic_field(rng, size) for _ in range(layers)])
    fields -= fields.max(axis=0, keepdims=True)
    weights = np.exp(np.float32(sharpness)*fields)
    weights /= weights.sum(axis=0, keepdims=True)
    return weights


def tile_to(image, size, tile_m, texture_m):
    """Repeat a texture at its physical scale across one tile of `tile_m` metres."""
    import cv2
    reps = max(1, round(tile_m/texture_m))
    cell = max(1, size//reps)
    small = cv2.resize(image, (cell, cell), interpolation=cv2.INTER_AREA)
    tiled = np.tile(small, (reps, reps, 1) if small.ndim == 3 else (reps, reps))
    if tiled.shape[0] != size:
        tiled = cv2.resize(tiled, (size, size), interpolation=cv2.INTER_LINEAR)
    return tiled


def composite_ground(bundle, catalog, terrain_config, seed, folder):
    """Write ground_{albedo,normal,roughness}.jpg into `folder`; return the recipe."""
    import cv2
    rng = np.random.RandomState(9100+seed)
    size, span = terrain_config['composite_px'], float(terrain_config['size_m'])
    count = min(len(catalog), terrain_config['ground_layers'])
    picks = sorted(int(i) for i in rng.choice(len(catalog), count, replace=False))
    # Unique across the whole patch, so a sand grain that repeats every couple
    # of metres does not also repeat as an identical 8 m tile.
    fields = np.stack([broad_field(rng, size) for _ in range(count)])
    fields -= fields.max(axis=0, keepdims=True)
    weights = np.exp(np.float32(6.0)*fields)
    weights /= weights.sum(axis=0, keepdims=True)
    albedo = np.zeros((size, size, 3), dtype=np.float32)
    normal = np.zeros((size, size, 3), dtype=np.float32)
    rough = np.zeros((size, size), dtype=np.float32)
    for weight, index in zip(weights, picks):
        layer = catalog[index]
        read = {key: cv2.imread(str(Path(bundle)/layer[key]), cv2.IMREAD_GRAYSCALE if key == 'roughness'
                                else cv2.IMREAD_COLOR) for key in ('albedo', 'normal', 'roughness')}
        if any(v is None for v in read.values()):
            raise FileNotFoundError(f'Missing ground texture for {layer["id"]}')
        w3 = weight[:, :, None]
        albedo += w3*tile_to(read['albedo'], size, span, layer['size_m']).astype(np.float32)
        normal += w3*(tile_to(read['normal'], size, span, layer['size_m']).astype(np.float32)/127.5-1)
        rough += weight*tile_to(read['roughness'], size, span, layer['size_m']).astype(np.float32)
    # Pull every layer toward one Martian hue while keeping its own light/dark
    # detail. Only colour variation along the Mars hue axis survives, so moss
    # or other off-palette tones in the source scans cannot show through.
    strength = float(rng.uniform(*terrain_config['tint_strength']))
    hue = MARS_RGB*rng.uniform(.9, 1.1, 3)
    tint_bgr = (hue/hue.mean())[::-1].astype(np.float32)
    luminance = albedo.mean(axis=2, keepdims=True)
    axis = tint_bgr-tint_bgr.mean()
    axis /= np.linalg.norm(axis)
    warmth = ((albedo-luminance)@axis)[:, :, None]*axis
    albedo = (1-strength)*(luminance+warmth) + strength*luminance*tint_bgr
    # The tint keeps colour and removes the sand's own contrast. Put the
    # fine grain back, and leave the broad shading to the line below.
    flat = luminance[:, :, 0]
    low = cv2.GaussianBlur(flat, (0, 0), max(1.0, size/120))
    albedo += (flat-low)[:, :, None]*0.85
    # A second, slower brightness change, several metres across, so the sand's
    # own repeat is not the largest pattern in the frame.
    shade = broad_field(rng, size, terms=4, max_cycles=2.0)
    shade = (shade-shade.min())/(np.ptp(shade)+1e-6)
    albedo *= (0.78+0.28*shade)[:, :, None]
    normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-6)
    quality = [cv2.IMWRITE_JPEG_QUALITY, 92]
    folder = Path(folder)
    cv2.imwrite(str(folder/'ground_albedo.jpg'), np.clip(albedo, 0, 255).astype(np.uint8), quality)
    cv2.imwrite(str(folder/'ground_normal.jpg'), np.clip((normal+1)*127.5, 0, 255).astype(np.uint8), quality)
    cv2.imwrite(str(folder/'ground_roughness.jpg'), np.clip(rough, 0, 255).astype(np.uint8), quality)
    dust_bgr = np.clip(albedo, 0, 255).mean(axis=(0, 1))
    return {'layers': [catalog[i]['id'] for i in picks], 'tint_strength': strength,
            'tint_rgb': hue.tolist(), 'span_m': span, 'pixels': size,
            'dust_bgr': dust_bgr.tolist()}
