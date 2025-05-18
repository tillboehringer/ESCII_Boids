import taichi as ti
import math
import cv2
import numpy as np
import os

ti.init(arch=ti.gpu)

# Settings
WIDTH, HEIGHT = 1600, 1000
NUM_BOIDS = 50000
MAX_SPEED = 4.0
NEIGHBOR_RADIUS = 25.0
SEPARATION_RADIUS = 12.0
TOTAL_FRAMES = 2000  # video duration in frames (e.g. 300 = 10s at 30 FPS)
OUTPUT_DIR = "boid_frames"
VIDEO_FILENAME = "boids_simulation.mp4"
FPS = 30

alignment_weight = 1.8
cohesion_weight = 4
separation_weight = 0.3

CELL_SIZE = NEIGHBOR_RADIUS * 2
GRID_SIZE_X = int(WIDTH // CELL_SIZE) + 1
GRID_SIZE_Y = int(HEIGHT // CELL_SIZE) + 1
MAX_BOIDS_PER_CELL = 100

positions = ti.Vector.field(2, dtype=ti.f32, shape=NUM_BOIDS)
velocities = ti.Vector.field(2, dtype=ti.f32, shape=NUM_BOIDS)
grid = ti.field(dtype=ti.i32, shape=(GRID_SIZE_X, GRID_SIZE_Y, MAX_BOIDS_PER_CELL))
grid_counts = ti.field(dtype=ti.i32, shape=(GRID_SIZE_X, GRID_SIZE_Y))


@ti.kernel
def initialize():
    radius = min(WIDTH, HEIGHT) * 0.2
    center = ti.Vector([WIDTH / 2, HEIGHT / 2])
    for i in range(NUM_BOIDS):
        r = ti.sqrt(ti.random()) * radius
        theta = ti.random() * 2 * math.pi
        pos = ti.Vector([r * ti.cos(theta), r * ti.sin(theta)]) + center
        positions[i] = pos
        angle = ti.random() * 2 * math.pi
        velocities[i] = ti.Vector([ti.cos(angle), ti.sin(angle)]) * MAX_SPEED


@ti.kernel
def build_grid():
    for i, j in grid_counts:
        grid_counts[i, j] = 0
    for i in range(NUM_BOIDS):
        pos = positions[i]
        cell_x = min(max(int(pos.x // CELL_SIZE), 0), GRID_SIZE_X - 1)
        cell_y = min(max(int(pos.y // CELL_SIZE), 0), GRID_SIZE_Y - 1)
        count = grid_counts[cell_x, cell_y]
        if count < MAX_BOIDS_PER_CELL:
            grid[cell_x, cell_y, count] = i
            grid_counts[cell_x, cell_y] = count + 1


@ti.func
def toroidal_wrap(offset):
    if offset.x > WIDTH / 2: offset.x -= WIDTH
    if offset.x < -WIDTH / 2: offset.x += WIDTH
    if offset.y > HEIGHT / 2: offset.y -= HEIGHT
    if offset.y < -HEIGHT / 2: offset.y += HEIGHT
    return offset


@ti.kernel
def update():
    for i in range(NUM_BOIDS):
        pos_i = positions[i]
        vel_i = velocities[i]

        align = ti.Vector([0.0, 0.0])
        cohere = ti.Vector([0.0, 0.0])
        separate = ti.Vector([0.0, 0.0])
        count = 0
        count_sep = 0

        cell_x = min(max(int(pos_i.x // CELL_SIZE), 0), GRID_SIZE_X - 1)
        cell_y = min(max(int(pos_i.y // CELL_SIZE), 0), GRID_SIZE_Y - 1)

        for dx in ti.static(range(-2, 3)):
            for dy in ti.static(range(-2, 3)):
                nx = cell_x + dx
                ny = cell_y + dy
                if 0 <= nx < GRID_SIZE_X and 0 <= ny < GRID_SIZE_Y:
                    for idx in range(grid_counts[nx, ny]):
                        j = grid[nx, ny, idx]
                        if i == j:
                            continue
                        pos_j = positions[j]
                        offset = toroidal_wrap(pos_j - pos_i)
                        dist = offset.norm()

                        if dist < NEIGHBOR_RADIUS:
                            align += velocities[j]
                            cohere += pos_j
                            count += 1
                            if dist < SEPARATION_RADIUS:
                                separate += (-offset.normalized() / dist) if dist > 1e-5 else ti.Vector([ti.random() - 0.5, ti.random() - 0.5])
                                count_sep += 1

        acc = ti.Vector([0.0, 0.0])

        if count > 0:
            align = (align.normalized() * MAX_SPEED - vel_i) if align.norm() > 1e-5 else ti.Vector([0.0, 0.0])
            acc += alignment_weight * align * 0.01

            center = cohere / count
            desired = toroidal_wrap(center - pos_i)
            desired = (desired.normalized() * MAX_SPEED - vel_i) if desired.norm() > 1e-5 else ti.Vector([0.0, 0.0])
            acc += cohesion_weight * desired * 0.005

        if count_sep > 0:
            separate = (separate.normalized() * MAX_SPEED - vel_i) if separate.norm() > 1e-5 else ti.Vector([0.0, 0.0])
            acc += separation_weight * separate * 0.15

        vel_i += acc
        if vel_i.norm() > MAX_SPEED:
            vel_i = vel_i.normalized() * MAX_SPEED

        velocities[i] = vel_i
        positions[i] = (pos_i + vel_i) % ti.Vector([WIDTH, HEIGHT])


# Prepare
initialize()
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Video writer
video_path = os.path.join(OUTPUT_DIR, VIDEO_FILENAME)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (WIDTH, HEIGHT))

# Main loop
for frame in range(TOTAL_FRAMES):
    build_grid()
    update()

    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    pos_np = positions.to_numpy().astype(np.int32)
    for p in pos_np:
        x, y = p
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            img[y, x] = (255, 255, 255)  # White boids

    video_writer.write(img)
    print(f"Rendering frame {frame + 1}/{TOTAL_FRAMES} ({(frame + 1) * 100 // TOTAL_FRAMES}%)")

video_writer.release()
print(f"Video saved to {video_path}")
