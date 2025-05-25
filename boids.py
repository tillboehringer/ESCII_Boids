import pygame
import pygame_widgets
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
from pygame_widgets.toggle import Toggle
import random
import math
from scipy.spatial import KDTree
from parameters import *
import numpy as np
from matplotlib import pyplot as plt
import csv



class Wolf:
    def __init__(self, pos, angle, ID, kind=0, saturation=init_sat_wolf):
        self.position = pygame.math.Vector2(*pos)
        self.velocity = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * MAX_SPEED
        self.acceleration = pygame.math.Vector2()
        self.kind = kind  # 0 = Predator, 1 = Prey
        self.saturation = saturation
        self.ID = ID

    def update(self):
        self.velocity += self.acceleration
        if self.velocity.length() > MAX_SPEED:
            self.velocity.scale_to_length(MAX_SPEED)
        self.position += self.velocity
        self.acceleration *= 0
        self.position.x %= WIDTH
        self.position.y %= HEIGHT

    def eat(self, boids):
        eaten_sheep_IDs = []
        for i, b in enumerate(boids):
            dist = self.position.distance_to(b.position)
            if dist <= EAT_RADIUS:
                self.saturation += wolf_eat_food
                eaten_sheep_IDs.append(b.ID)
        return eaten_sheep_IDs

    def apply_behaviors(self, neighbors):
        align = self.align(neighbors) * alignment_weight[0]
        cohere = self.cohere(neighbors) * cohesion_weight[0]
        separate = self.separate(neighbors) * separation_weight[0]
        self.acceleration += align + cohere + separate

    def apply_hunt(self, prey):
        hunt = self.hunt(prey) * predator_weight[0]
        self.acceleration += hunt

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

    def get_color(self):
        if self.kind == 0:
            return 255, 0, 0
        elif self.kind == 1:
            return 0, 255, 0
        else:
            return 255, 255, 255

    def draw(self, screen):
        direction = self.velocity.normalize() * 10
        pygame.draw.aaline(screen, self.get_color(), (self.position.x, self.position.y),
                           (self.position.x + direction.x, self.position.y + direction.y))
        pygame.draw.circle(screen, self.get_color(), (self.position.x, self.position.y), 4)


class Sheep:
    def __init__(self, pos, angle, ID, kind = 1, saturation=init_sat_sheep):
        self.position = pygame.math.Vector2(*pos)
        self.velocity = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * MAX_SPEED
        self.acceleration = pygame.math.Vector2()
        self.kind = kind # 0 = Predator, 1 = Prey
        self.saturation = saturation
        self.ID = ID

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

    def apply_flee(self, predator):
        flee = self.flee(predator) * prey_weight[0]
        self.acceleration += flee

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
        if self.kind == 0:
            return (255, 0, 0)
        elif self.kind == 1:
            return (0, 255, 0)
        else:
            return (255, 255, 255)

    def draw(self, screen):
        direction = self.velocity.normalize() * 10
        pygame.draw.aaline(screen, self.get_color(), (self.position.x, self.position.y), (self.position.x + direction.x, self.position.y + direction.y))
        pygame.draw.circle(screen, self.get_color(), (self.position.x, self.position.y), 4)

# --- Main Loop ---

NUM_FRAMES_GRAPH = 180
x = np.arange(NUM_FRAMES_GRAPH)

num_sheep = NUM_PREY * np.ones(NUM_FRAMES_GRAPH)
num_wolves = NUM_PREDATORS * np.ones(NUM_FRAMES_GRAPH)

fig, ax = plt.subplots()
ax.plot(x, num_sheep)
ax.plot(x, num_wolves)
ax.set_ylim((0, 300))
# ax.set_xbound((0, NUM_FRAMES_GRAPH))
fig.show()

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Boids")
clock = pygame.time.Clock()


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
predator_slider = Slider(screen, 170, 50, 150, 10, min=0, max=3, step=0.1, initial=1, handleColour=(255, 0, 0))

predator_radius_text = TextBox(screen, 170, 80, 150, 30, fontSize=15)
predator_radius_text.disable()
predator_radius_slider = Slider(screen, 170, 100, 150, 10, min=1, max=300, step=1, initial=100, handleColour=(255, 0, 0))

prey_text = TextBox(screen, 170, 130, 150, 30, fontSize=15)
prey_text.disable()
prey_slider = Slider(screen, 170, 150, 150, 10, min=0, max=3, step=0.1, initial=1, handleColour=(0, 255, 0))

prey_radius_text = TextBox(screen, 170, 180, 150, 30, fontSize=15)
prey_radius_text.disable()
prey_radius_slider = Slider(screen, 170, 200, 150, 10, min=1, max=300, step=1, initial=100, handleColour=(0, 255, 0))


# boids = [Boid(pos=(random.uniform(0, WIDTH), random.uniform(0, HEIGHT)), angle = random.uniform(0, 2 * math.pi), kind=random.randint(0,1)) for _ in range(NUM_BOIDS)]
wolves = [Wolf(pos=(random.normalvariate(mu=WIDTH/4, sigma=WIDTH/20), random.normalvariate(mu=HEIGHT/2, sigma=HEIGHT/20)),
               angle = random.uniform(0, 2 * math.pi), ID=i, saturation=np.random.randint(5e6, 5e8)) for i in range(NUM_PREDATORS)]
sheep = [Sheep(pos=(random.normalvariate(mu=WIDTH*3/4, sigma=WIDTH/20), random.normalvariate(mu=HEIGHT/2, sigma=HEIGHT/20)),
               angle = random.uniform(0, 2 * math.pi), ID=i+NUM_PREDATORS, saturation=np.random.randint(5e2, 1e5)) for i in range(NUM_PREY)]
boids = []
boids.extend(wolves)
boids.extend(sheep)
ID_COUNT += len(boids)


running = True
loop_counter = 0
while running:
    screen.fill((30, 30, 30))
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            running = False

    points = [(boid.position.x, boid.position.y) for boid in boids]
    tree = KDTree(points)

    RADIUS = max(NEIGHBOR_RADIUS, SEPARATION_RADIUS, PREDATOR_RADIUS, PREY_RADIUS)
    num_wolves[-1] = 0
    num_sheep[-1] = 0
    # in this loop: look at distances and change velocity, position etc
    for i, boid in enumerate(boids):
        if on_off_toggle.getValue():
            idx = tree.query_ball_point(points[i], RADIUS)
            # only takes into account neighbours of same kind:
            neighbors = [boids[j] for j in idx if j != i and boids[j].kind == boid.kind]
            if boid.kind == 0: # if wolf:
                num_wolves[-1] += 1
                prey = [boids[j] for j in idx if boids[j].kind == 1]
                boid.apply_hunt(prey)
            if boid.kind == 1:
                num_sheep[-1] += 1
                predator = [boids[j] for j in idx if boids[j].kind == 0]
                boid.apply_flee(predator)
            boid.apply_behaviors(neighbors)
            boid.update()
        boid.draw(screen)

    # here: look at new neighbours and apply eating and reproduction
    all_dead = []
    wolf_change = 0
    sheep_change = 0
    for i, boid in enumerate(boids):
        dead_sheep_IDs = []
        dead_wolf_IDs = []
        if boid.ID in dead_sheep_IDs:
            continue
        points = [(boid.position.x, boid.position.y) for boid in boids]
        idx = tree.query_ball_point(points[i], RADIUS)
        # only takes into account neighbours of the other kind:
        neighbors = [boids[j] for j in idx if j != i and boids[j].kind != boid.kind]
        if boid.kind == 1: # if sheep:
            # check for reproduction
            if boid.saturation > sheep_reproduction_sat:
                boids.append(Sheep(pos=boid.position,
                                  angle=random.uniform(0, 2 * math.pi), ID=ID_COUNT))
                ID_COUNT += 1
                boid.saturation -= sheep_reproduction_loss
            boid.saturation += sheep_eats
        if boid.kind == 0: # if wolf:
            # first eat sheep
            sheep_dead = boid.eat(neighbors)
            for x in sheep_dead[-1:]:
                if x in dead_sheep_IDs:
                    sheep_dead.remove(x)
            dead_sheep_IDs.extend(sheep_dead)
            # then check saturation of wolf for reproduction
            if boid.saturation > wolf_reproduction_sat:
                boids.append(Wolf(pos=boid.position,
                                   angle = random.uniform(0, 2 * math.pi), ID=ID_COUNT))
                ID_COUNT += 1
                boid.saturation -= wolf_reproduction_loss
            # then check for saturation for dying
            if boid.saturation < wolf_death:
                dead_wolf_IDs.append(boid.ID)
            boid.saturation -= wolf_no_food


        all_dead.extend(dead_wolf_IDs)
        all_dead.extend(dead_sheep_IDs)
    #exit loop and change the amount of sheep and wolves

    all_dead.sort()

    for boid in boids:
        if boid.ID in all_dead[::-1]:
            boids.remove(boid)


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

    num_sheep = np.append(num_sheep, num_sheep[-1])[-NUM_FRAMES_GRAPH:]
    num_wolves = np.append(num_wolves, num_wolves[-1])[-NUM_FRAMES_GRAPH:]
    x = np.arange(NUM_FRAMES_GRAPH)

    if loop_counter % 10 == 0 and on_off_toggle.getValue():
        ax.clear()
        ax.plot(x, num_sheep, label='sheep')
        ax.plot(x, num_wolves, label='wolves')
        ax.legend()
        fig.canvas.draw()
        # Here is the option to track pop size and save
        # with open(f"popsize.csv", 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow([float(num_sheep[-1]), float(num_wolves[-1])])


    pygame_widgets.update(events)
    pygame.display.flip()
    clock.tick(60)
    loop_counter += 1



pygame.quit()

