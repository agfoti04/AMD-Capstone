from four_wheel_sim.maze import WALLS

RES = 0.05
OX, OY, W, H = -1.0, -1.0, 120, 80    # origin (m), size in pixels (6 m x 4 m)

grid = [[254] * W for _ in range(H)]  # 254 = free, 0 = occupied
for cx, cy, sx, sy in WALLS:
    for row in range(H):
        for col in range(W):
            x, y = OX + (col + 0.5) * RES, OY + (H - row - 0.5) * RES
            if abs(x - cx) <= sx / 2 and abs(y - cy) <= sy / 2:
                grid[row][col] = 0

with open('maze.pgm', 'wb') as f:
    f.write(f'P5 {W} {H} 255\n'.encode())
    for r in grid:
        f.write(bytes(r))

with open('maze.yaml', 'w') as f:
    f.write(f'image: maze.pgm\nresolution: {RES}\norigin: [{OX}, {OY}, 0.0]\n'
            'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n')