import pygame
import pygame_widgets
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
from pygame_widgets.toggle import Toggle
import random
import math
from scipy.spatial import KDTree

WIDTH, HEIGHT = 1000, 800
NUM_BOIDS = 150
MAX_SPEED = 4
NEIGHBOR_RADIUS = 50
SEPARATION_RADIUS = 20

alignment_weight = [1.0]
cohesion_weight = [1.0]
separation_weight = [1.0]

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
        align = self.align(neighbors) * alignment_weight[0]
        cohere = self.cohere(neighbors) * cohesion_weight[0]
        separate = self.separate(neighbors) * separation_weight[0]
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
            if dist < SEPARATION_RADIUS and dist > 0:
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
        direction = self.velocity.normalize() * 10
        pygame.draw.aaline(screen, (255, 255, 255), (self.position.x, self.position.y), (self.position.x + direction.x, self.position.y + direction.y))
        pygame.draw.circle(screen, (255, 255, 255), (self.position.x, self.position.y), 4)

# --- Main Loop ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Boids")
    clock = pygame.time.Clock()

    alignment_text = TextBox(screen, 10, 10, 150, 30, fontSize=15)
    alignment_text.disable()
    alignment_slider = Slider(screen, 10, 30, 150, 10, min=0, max=3, step=0.1, initial=1)

    cohesion_text = TextBox(screen, 10, 50, 150, 30, fontSize=15)
    cohesion_text.disable()
    cohesion_slider = Slider(screen, 10, 70, 150, 10, min=0, max=3, step=0.1, initial=1)

    separation_text = TextBox(screen, 10, 90, 150, 30, fontSize=15)
    separation_text.disable()
    separation_slider = Slider(screen, 10, 110, 150, 10, min=0, max=3, step=0.1, initial=1)

    on_off_toggle = Toggle(screen, 10, 130, 30, 10, startOn = True)

    boids = [Boid() for _ in range(NUM_BOIDS)]

    running = True
    while running:
        screen.fill((30, 30, 30))
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False

        points = [(b.position.x, b.position.y) for b in boids]
        tree = KDTree(points)

        for i, boid in enumerate(boids):
            if on_off_toggle.getValue():
                idx = tree.query_ball_point(points[i], NEIGHBOR_RADIUS)
                neighbors = [boids[j] for j in idx if j != i]
                boid.apply_behaviors(neighbors)
                boid.update()
            boid.draw(screen)

        global alignment_weight
        global cohesion_weight
        global separation_weight

        alignment_weight = [alignment_slider.getValue()]
        cohesion_weight = [cohesion_slider.getValue()]
        separation_weight = [separation_slider.getValue()]

        alignment_text.setText(f'Alignment {alignment_weight[0]:.1f}')
        cohesion_text.setText(f'Cohesion {cohesion_weight[0]:.1f}')
        separation_text.setText(f'Separation {separation_weight[0]:.1f}')

        pygame_widgets.update(events)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
