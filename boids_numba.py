"""
boids_numba.py

This file implements a boid simulation using the Numba library to accelerate computations.

Numba is a Just-In-Time (JIT) compiler for Python that translates a subset of Python and NumPy code into fast machine code.
It is particularly well-suited for numerical computations and loops, making it ideal for simulations like this one.

In this project, Numba is used to:
- Accelerate the update logic for boids by compiling the `update_boids` function to machine code.
- Enable parallel execution of loops using the `prange` construct, which distributes work across multiple CPU cores.

The result is a significant performance improvement, allowing the simulation to handle thousands of boids in real-time.
"""

import pygame
import random
import math
import numpy as np
from numba import njit, prange

# 1000 Boids: ~60 Fps
# 5000 Boids: ~3 Fps

# Settings
WIDTH, HEIGHT = 1000, 800
NUM_BOIDS = 1000
MAX_SPEED = 4.0
NEIGHBOR_RADIUS = 50
SEPARATION_RADIUS = 20

inertia = 0.9

alignment_weight = 1.0
cohesion_weight = 1.0
separation_weight = 0.5


@njit(parallel=True)
def update_boids(positions, velocities):
    """
    This function calculates the new velocities for all boids based on three main behaviors:
    Numba's @njit decorator with parallel=True enables Just-In-Time (JIT) compilation and parallel execution,
    significantly improving performance for large arrays and loops. This is crucial for real-time simulation
    with thousands of boids.
    """
    new_velocities = np.zeros_like(velocities)
    for i in prange(len(positions)):  # prange allows parallel execution
        px, py = positions[i]
        vx, vy = velocities[i]

        align = np.zeros(2)  # Alignment vector
        cohere = np.zeros(2)  # Cohesion vector
        separate = np.zeros(2)  # Separation vector

        count = 0  # Number of neighbors for alignment and cohesion
        count_sep = 0  # Number of neighbors for separation

        for j in range(len(positions)):
            if i == j:
                continue
            dx = positions[j][0] - px
            dy = positions[j][1] - py

            # Toroidal wrapping ensures boids interact across screen edges
            dx = dx - WIDTH if dx > WIDTH / 2 else dx + WIDTH if dx < -WIDTH / 2 else dx
            dy = dy - HEIGHT if dy > HEIGHT / 2 else dy + HEIGHT if dy < -HEIGHT / 2 else dy

            dist = math.hypot(dx, dy)  # Euclidean distance
            if dist < NEIGHBOR_RADIUS:
                # Alignment: Sum velocities of neighbors
                align += velocities[j]
                # Cohesion: Sum positions of neighbors
                cohere += positions[j]
                count += 1
                if dist < SEPARATION_RADIUS:
                    # Separation: Move away from neighbors within a smaller radius
                    if dist > 1e-3:  # Avoid division by zero
                        diff = np.array([-dx, -dy]) / dist
                    else:
                        diff = np.random.uniform(-1, 1, 2)  # Random direction if too close
                    separate += diff
                    count_sep += 1

        acceleration = np.zeros(2)

        if count > 0:
            # Alignment: Steer towards the average velocity of neighbors
            align /= count
            mag = math.hypot(align[0], align[1])
            if mag > 0:
                align = (align / mag) * MAX_SPEED - np.array([vx, vy])
                align *= 0.05  # Scale alignment force

            # Cohesion: Steer towards the center of mass of neighbors
            center = cohere / count
            dx = center[0] - px
            dy = center[1] - py
            dx = dx - WIDTH if dx > WIDTH / 2 else dx + WIDTH if dx < -WIDTH / 2 else dx
            dy = dy - HEIGHT if dy > HEIGHT / 2 else dy + HEIGHT if dy < -HEIGHT / 2 else dy
            desired = np.array([dx, dy])
            mag = math.hypot(desired[0], desired[1])
            if mag > 0:
                desired = (desired / mag) * MAX_SPEED - np.array([vx, vy])
                cohere = desired * 0.01  # Scale cohesion force
            else:
                cohere = np.zeros(2)
        else:
            align = np.zeros(2)
            cohere = np.zeros(2)

        if count_sep > 0:
            # Separation: Steer away from nearby boids
            separate /= count_sep
            mag = math.hypot(separate[0], separate[1])
            if mag > 0:
                separate = (separate / mag) * MAX_SPEED - np.array([vx, vy])
                separate *= 0.25  # Scale separation force
        else:
            separate = np.zeros(2)

        # Combine the three behaviors with respective weights
        acceleration = alignment_weight * align + cohesion_weight * cohere + separation_weight * separate
        vx += acceleration[0]
        vy += acceleration[1]

        # Apply inertia to smooth velocity changes
        vx = inertia * vx + (1 - inertia) * velocities[i][0]
        vy = inertia * vy + (1 - inertia) * velocities[i][1]

        # Limit speed to MAX_SPEED
        speed = math.hypot(vx, vy)
        if speed > MAX_SPEED:
            vx = (vx / speed) * MAX_SPEED
            vy = (vy / speed) * MAX_SPEED

        new_velocities[i][0] = vx
        new_velocities[i][1] = vy

    return new_velocities

def main():
    """
    The main function initializes the simulation, handles rendering, and updates boid positions and velocities.
    Pygame is used for visualization, while the update_boids function handles the physics of the simulation.
    """
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Boids with Numba Acceleration")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    # Initialize boid positions and velocities
    positions = np.array([[random.uniform(0, WIDTH), random.uniform(0, HEIGHT)] for _ in range(NUM_BOIDS)], dtype=np.float32)
    velocities = np.array([[math.cos(random.uniform(0, 2 * math.pi)) * MAX_SPEED,
                            math.sin(random.uniform(0, 2 * math.pi)) * MAX_SPEED] for _ in range(NUM_BOIDS)], dtype=np.float32)

    running = True
    while running:
        screen.fill((30, 30, 30))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update boid velocities and positions
        velocities = update_boids(positions, velocities)
        positions += velocities

        # Toroidal wrapping to keep boids within screen bounds
        positions[:, 0] %= WIDTH
        positions[:, 1] %= HEIGHT

        # Draw boids
        for pos in positions:
            pygame.draw.circle(screen, (255, 255, 255), (int(pos[0]), int(pos[1])), 2)

        # Display FPS
        fps = clock.get_fps()
        fps_text = font.render(f"FPS: {fps:.2f}", True, (255, 255, 0))
        screen.blit(fps_text, (10, 10))

        pygame.display.flip()
        clock.tick(0)  # Unlimited FPS

    pygame.quit()

if __name__ == "__main__":
    main()