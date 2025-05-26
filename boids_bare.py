import pygame
import random
import math
from scipy.spatial import KDTree

# ~20 FPS

# Simulation settings
WIDTH, HEIGHT = 1000, 800
NUM_BOIDS = 1000
MAX_SPEED = 4
NEIGHBOR_RADIUS = 50
SEPARATION_RADIUS = 20

# Boid behavior weights
alignment_weight = 1.0
cohesion_weight = 2.5
separation_weight = 1.0

class Boid:
    def __init__(self):
        self.position = pygame.math.Vector2(random.uniform(0, WIDTH), random.uniform(0, HEIGHT))
        angle = random.uniform(0, 2 * math.pi)
        self.velocity = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * MAX_SPEED
        self.acceleration = pygame.math.Vector2()

    def update(self):
        self.velocity += self.acceleration
        if self.velocity.length() > MAX_SPEED:
            self.velocity.scale_to_length(MAX_SPEED)
        self.position += self.velocity
        self.acceleration *= 0
        self.position.x %= WIDTH
        self.position.y %= HEIGHT

    def apply_behaviors(self, neighbors):
        align = self.align(neighbors) * alignment_weight
        cohere = self.cohere(neighbors) * cohesion_weight
        separate = self.separate(neighbors) * separation_weight
        self.acceleration += align + cohere + separate

    def align(self, boids):
        steering = pygame.math.Vector2()
        total = len(boids)
        if total > 0:
            for b in boids:
                steering += b.velocity
            steering /= total
            steering = steering.normalize() * MAX_SPEED - self.velocity
        return steering * 0.05

    def cohere(self, boids):
        steering = pygame.math.Vector2()
        total = len(boids)
        if total > 0:
            center = sum((b.position for b in boids), pygame.math.Vector2())
            center /= total
            desired = center - self.position
            desired = desired.normalize() * MAX_SPEED
            steering = desired - self.velocity
        return steering * 0.01

    def separate(self, boids):
        steering = pygame.math.Vector2()
        total = 0
        for b in boids:
            dist = self.position.distance_to(b.position)
            if 0 < dist < SEPARATION_RADIUS:
                diff = self.position - b.position
                diff /= dist
                steering += diff
                total += 1
        if total > 0:
            steering /= total
            if steering.length() > 0:
                steering = steering.normalize() * MAX_SPEED - self.velocity
        return steering * 0.1

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 255, 255), (int(self.position.x), int(self.position.y)), 3)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Boids (FPS Measured)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    boids = [Boid() for _ in range(NUM_BOIDS)]

    running = True
    while running:
        screen.fill((30, 30, 30))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        points = [(b.position.x, b.position.y) for b in boids]
        tree = KDTree(points)

        for i, boid in enumerate(boids):
            idx = tree.query_ball_point(points[i], NEIGHBOR_RADIUS)
            neighbors = [boids[j] for j in idx if j != i]
            boid.apply_behaviors(neighbors)
            boid.update()
            boid.draw(screen)

        # FPS display
        fps = clock.get_fps()
        fps_text = font.render(f"FPS: {fps:.2f}", True, (255, 255, 0))
        screen.blit(fps_text, (10, 10))

        pygame.display.flip()
        clock.tick(0)  # No cap to measure raw FPS

    pygame.quit()

if __name__ == "__main__":
    main()
