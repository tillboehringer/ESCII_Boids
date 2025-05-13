import pygame
import pygame_widgets
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
from pygame_widgets.toggle import Toggle
import random
import math
from scipy.spatial import KDTree

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

import icecream as ic

WIDTH, HEIGHT = 1200, 1200
MAX_SPEED = 4
NEIGHBOR_RADIUS = 50
SEPARATION_RADIUS = 20
PREDATOR_RADIUS = 100
PREY_RADIUS = 60
CHANGE_RADIUS = 5

alignment_weight = [1.0]
cohesion_weight = [1.0]
separation_weight = [1.0]
predator_weight = [1.2]
prey_weight = [0.6]

KINDS_NUM = 5
NUM_BOIDS_PER_KIND = 60

cmap = mpl.colormaps['jet']
colors_rgba = cmap(np.linspace(0, 1, KINDS_NUM))
# print(colors_rgba)
colors = [tuple(int(c*255) for c in color) for color in colors_rgba[:, :3]]
# print(colors)

class Boid:
    def __init__(self, pos, angle, kind = 0):
        self.position = pygame.math.Vector2(*pos)
        self.velocity = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * MAX_SPEED
        self.acceleration = pygame.math.Vector2()
        self.kind = kind # 0 = Predator, 1 = Prey

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

    def apply_hunt(self, prey):
        hunt = self.hunt(prey) * predator_weight[0]
        self.acceleration += hunt

    def apply_flee(self, predator):
        flee = self.flee(predator) * prey_weight[0]
        self.acceleration += flee

    def change_kind(self, boids):
        for b in boids:
            dist = self.position.distance_to(b.position)
            if dist <= CHANGE_RADIUS:
                self.kind = b.kind
        
    def align(self, boids):
        steering = pygame.math.Vector2()
        total = 0
        for b in boids:
            dist = self.position.distance_to(b.position)
            if dist < NEIGHBOR_RADIUS and dist > 0:
                steering += b.velocity
                total += 1
        if total > 0:
            steering /= total
            steering = steering.normalize() * MAX_SPEED - self.velocity
        return steering * 0.05

    def cohere(self, boids):
        steering = pygame.math.Vector2()
        center = pygame.math.Vector2()
        total = 0
        for b in boids:
            dist = self.position.distance_to(b.position)
            if dist < NEIGHBOR_RADIUS and dist > 0:
                center += b.position
                total += 1
        if total > 0:
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

    def hunt(self, boids):
        steering = pygame.math.Vector2()
        center = pygame.math.Vector2()
        total = 0
        for b in boids:
            dist = self.position.distance_to(b.position)
            if dist < PREDATOR_RADIUS and dist > 0:
                center += b.position
                total += 1
        if total > 0:
            center /= total
            desired = center - self.position
            desired = desired.normalize() * MAX_SPEED
            steering = desired - self.velocity
        return steering * 0.1
    
    def flee(self, boids):
        steering = pygame.math.Vector2()
        center = pygame.math.Vector2()
        total = 0
        for b in boids:
            dist = self.position.distance_to(b.position)
            if dist < PREY_RADIUS and dist > 0:
                center += b.position
                total += 1
        if total > 0:
            center /= total
            desired = center - self.position
            desired = desired.normalize() * MAX_SPEED
            steering = desired - self.velocity
        return - steering * 0.1

    def get_color(self):
        return colors[self.kind]

    def draw(self, screen):
        direction = self.velocity.normalize() * 10
        pygame.draw.aaline(screen, self.get_color(), (self.position.x, self.position.y), (self.position.x + direction.x, self.position.y + direction.y))
        pygame.draw.circle(screen, self.get_color(), (self.position.x, self.position.y), 4)

# --- Main Loop ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Boids")
    clock = pygame.time.Clock()
    
    global alignment_weight
    global cohesion_weight
    global separation_weight
    global NEIGHBOR_RADIUS
    global SEPARATION_RADIUS
    global PREDATOR_RADIUS
    global PREY_RADIUS
    global prey_weight
    global predator_weight
    global boids_count

    on_off_toggle = Toggle(screen, 10, 10, 30, 10, startOn = False)

    frames_text = TextBox(screen, 50, 5, 90, 30, fontSize=15)
    frames_text.disable()

    alignment_text = TextBox(screen, 10, 30, 150, 30, fontSize=15)
    alignment_text.disable()
    alignment_slider = Slider(screen, 10, 50, 150, 10, min=0, max=3, step=0.1, initial=1)

    cohesion_text = TextBox(screen, 10, 80, 150, 30, fontSize=15)
    cohesion_text.disable()
    cohesion_slider = Slider(screen, 10, 100, 150, 10, min=0, max=3, step=0.1, initial=1)

    separation_text = TextBox(screen, 10, 130, 150, 30, fontSize=15)
    separation_text.disable()
    separation_slider = Slider(screen, 10, 150, 150, 10, min=0, max=3, step=0.1, initial=1)

    neighbor_radius_text = TextBox(screen, 10, 180, 150, 30, fontSize=15)
    neighbor_radius_text.disable()
    neighbor_radius_slider = Slider(screen, 10, 200, 150, 10, min=1, max=200, step=1, initial=50)

    separation_radius_text = TextBox(screen, 10, 230, 150, 30, fontSize=15)
    separation_radius_text.disable()
    separation_radius_slider = Slider(screen, 10, 250, 150, 10, min=1, max=100, step=1, initial=20)

    predator_text = TextBox(screen, 170, 30, 150, 30, fontSize=15)
    predator_text.disable()
    predator_slider = Slider(screen, 170, 50, 150, 10, min=0, max=3, step=0.1, initial=1.2, handleColour=(255, 0, 0))

    predator_radius_text = TextBox(screen, 170, 80, 150, 30, fontSize=15)
    predator_radius_text.disable()
    predator_radius_slider = Slider(screen, 170, 100, 150, 10, min=1, max=300, step=1, initial=100, handleColour=(255, 0, 0))

    prey_text = TextBox(screen, 170, 130, 150, 30, fontSize=15)
    prey_text.disable()
    prey_slider = Slider(screen, 170, 150, 150, 10, min=0, max=3, step=0.1, initial=0.6, handleColour=(0, 255, 0))

    prey_radius_text = TextBox(screen, 170, 180, 150, 30, fontSize=15)
    prey_radius_text.disable()
    prey_radius_slider = Slider(screen, 170, 200, 150, 10, min=1, max=300, step=1, initial=60, handleColour=(0, 255, 0))

    change_radius_text = TextBox(screen, 170, 230, 150, 30, fontSize=15)
    change_radius_text.disable()
    change_radius_slider = Slider(screen, 170, 250, 150, 10, min=1, max=50, step=1, initial=5)

    boids = []

    for kind in range(KINDS_NUM):
        boids.extend([Boid(pos=(random.uniform(0, WIDTH), random.uniform(0, HEIGHT)), angle = random.uniform(0, 2 * math.pi), kind=kind) for _ in range(NUM_BOIDS_PER_KIND)])

    NUM_FRAMES_GRAPH = 180
    boids_count = np.zeros((KINDS_NUM, NUM_FRAMES_GRAPH))
    x = np.arange(NUM_FRAMES_GRAPH)

    fig, ax = plt.subplots()
    ax.stackplot(x, boids_count)
    ax.set_ybound((0,NUM_BOIDS_PER_KIND*KINDS_NUM))
    ax.set_xbound((0, NUM_FRAMES_GRAPH))
    fig.show()

    running = True
    loop_counter = 0
    while running:
        screen.fill((150, 150, 150))
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False

        points = [(b.position.x, b.position.y) for b in boids]
        tree = KDTree(points)

        RADIUS = max(NEIGHBOR_RADIUS, SEPARATION_RADIUS, PREDATOR_RADIUS, PREY_RADIUS)

        boids_num = np.zeros(KINDS_NUM)

        for i, boid in enumerate(boids):
            if on_off_toggle.getValue():
                idx = tree.query_ball_point(points[i], RADIUS)

                prey_kinds = [(boid.kind+n+1)%KINDS_NUM for n in range(int((KINDS_NUM-1)/2))]
                predator_kinds = [(boid.kind-n-1)%KINDS_NUM for n in range(int((KINDS_NUM-1)/2))]

                # print(boid.kind, prey_kinds, predator_kinds)

                neighbors = [boids[j] for j in idx if j != i and boids[j].kind == boid.kind]
                boid.apply_behaviors(neighbors)

                boids_num[boid.kind] += 1

                prey = [boids[j] for j in idx if boids[j].kind in prey_kinds]
                boid.apply_hunt(prey)

                predator = [boids[j] for j in idx if boids[j].kind in predator_kinds]
                boid.apply_flee(predator)
                boid.change_kind(predator)

                boid.update()
            boid.draw(screen)


        alignment_weight = [alignment_slider.getValue()]
        cohesion_weight = [cohesion_slider.getValue()]
        separation_weight = [separation_slider.getValue()]
        NEIGHBOR_RADIUS = neighbor_radius_slider.getValue()
        SEPARATION_RADIUS = separation_radius_slider.getValue()

        alignment_text.setText(f'Alignment {alignment_weight[0]:.1f}')
        cohesion_text.setText(f'Cohesion {cohesion_weight[0]:.1f}')
        separation_text.setText(f'Separation {separation_weight[0]:.1f}')
        neighbor_radius_text.setText(f'Neigh_radius {NEIGHBOR_RADIUS}')
        separation_radius_text.setText(f'Sep_radius {SEPARATION_RADIUS}')
        frames_text.setText(f'FPS {clock.get_fps():.1f}')

        predator_weight = [predator_slider.getValue()]
        predator_text.setText(f'Hunt {predator_weight[0]:.1f}')
        PREDATOR_RADIUS = predator_radius_slider.getValue()
        predator_radius_text.setText(f'Hunt_radius {PREDATOR_RADIUS}')

        prey_weight = [prey_slider.getValue()]
        prey_text.setText(f'Flee {prey_weight[0]:.1f}')
        PREY_RADIUS = prey_radius_slider.getValue()
        prey_radius_text.setText(f'Flee_radius {PREY_RADIUS}')

        CHANGE_RADIUS = change_radius_slider.getValue()
        change_radius_text.setText(f'Change_rad {CHANGE_RADIUS}')

        if on_off_toggle.getValue():
            boids_count = np.append(boids_count, np.array([boids_num]).T, axis=1)[:, -NUM_FRAMES_GRAPH:]
            print(boids_num, loop_counter)
        if loop_counter%20 == 0 and on_off_toggle.getValue():
            ax.clear()
            ax.stackplot(x, boids_count, colors=colors_rgba)
            fig.canvas.draw()


        pygame_widgets.update(events)
        pygame.display.flip()
        loop_counter += 1
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
