import taichi as ti
import math

""" To monitor AMD GPU Traffic: "sudo intel_gpu_top" """

# Taichi Documentation: https://docs.taichi-lang.org/docs/tutorial

# For the next improvement approach, we want to use the powerhouse of our computer - the GPU.
# Initialize Taichi with GPU support. Taichi is a high-performance computation framework that allows us to write Python-like code
# and execute it efficiently on the GPU. GPUs excel at parallel processing, making them ideal for simulations like this,
# where thousands of boids need to be updated simultaneously.
ti.init(arch=ti.gpu)



WIDTH, HEIGHT = 800, 600
# NUM_BOIDS = 10000 
NUM_BOIDS = 20000 
MAX_SPEED = 4.0  
NEIGHBOR_RADIUS = 50.0  
SEPARATION_RADIUS = 10.0 

alignment_weight = 1.5
cohesion_weight = 2.0
separation_weight = 0.4

# Grid parameters for spatial partitioning. Spatial partitioning divides the simulation space into a grid to reduce the number
# of pairwise distance calculations. This optimization is crutial has a great impact on the calculations needed to simulate the boids behaviour. 
CELL_SIZE = NEIGHBOR_RADIUS 
GRID_SIZE_X = int(WIDTH // CELL_SIZE) + 1 
GRID_SIZE_Y = int(HEIGHT // CELL_SIZE) + 1 
MAX_BOIDS_PER_CELL = 5000 

# Taichi fields (data structures) to store boid positions and velocities. These fields are allocated on the GPU for fast access.
positions = ti.Vector.field(2, dtype=ti.f32, shape=NUM_BOIDS)  # 2D positions of boids
velocities = ti.Vector.field(2, dtype=ti.f32, shape=NUM_BOIDS)  # 2D velocities of boids

# Grid data structures for spatial partitioning. The grid stores boid indices and the count of boids in each cell.
# These fields are allocated on the GPU to enable efficient parallel updates.

# From the taichi Doc: https://docs.taichi-lang.org/docs/field
# "Fields in Taichi are the global data containers, which can be accessed from both the Python scope and the Taichi scope. 
# Just like an ndarray in NumPy or a tensor in PyTorch, a field in Taichi is defined as a multi-dimensional array of elements, 
# and elements in a field can be a Scalar, a Vector, a Matrix, or a Struct."

# Our Taichi field is initialized with 32-bit integers
grid = ti.field(dtype=ti.i32, shape=(GRID_SIZE_X, GRID_SIZE_Y, MAX_BOIDS_PER_CELL))  # Grid storing boid indices
grid_counts = ti.field(dtype=ti.i32, shape=(GRID_SIZE_X, GRID_SIZE_Y))  # Count of boids in each grid cell

@ti.kernel
def initialize():
    """
    Initialize the positions and velocities of all boids. Positions are distributed in a circular region around the center
    of the screen, and velocities are assigned random directions with a fixed speed. This kernel runs on the GPU, with
    each boid's initialization computed in parallel.
    """
    radius = min(WIDTH, HEIGHT) * 0.2  # Radius of the circular region for initial positions
    center = ti.Vector([WIDTH / 2, HEIGHT / 2]) 
    for i in range(NUM_BOIDS):
        # Randomly distribute positions within the circular region
        r = ti.sqrt(ti.random()) * radius  # Radial distance (sqrt ensures uniform distribution in the circle by compensating for r² growth of area in circle)
        theta = ti.random() * 2 * math.pi  # Random angle
        pos = ti.Vector([r * ti.cos(theta), r * ti.sin(theta)]) + center
        positions[i] = pos

        # Assign random initial velocities with a fixed speed
        angle = ti.random() * 2 * math.pi
        velocities[i] = ti.Vector([ti.cos(angle), ti.sin(angle)]) * MAX_SPEED

@ti.kernel
def build_grid():
    """
    Build the spatial partitioning grid. This kernel resets the grid counts and assigns each boid to the appropriate grid cell
    based on its position. The grid allows us to limit neighbor searches to nearby cells, significantly reducing the number
    of distance calculations.
    """
    # Reset the grid counts to zero. This ensures that the grid is cleared before updating it with the current boid positions.
    for i, j in grid_counts:
        grid_counts[i, j] = 0

    # Assign each boid to a grid cell based on its position
    for i in range(NUM_BOIDS):
        pos = positions[i]
        cell_x = int(pos.x // CELL_SIZE)  # Determine the x-coordinate of the grid cell
        cell_y = int(pos.y // CELL_SIZE) 

        # Clamp the cell indices to ensure they are within the grid bounds
        cell_x = min(max(cell_x, 0), GRID_SIZE_X - 1)
        cell_y = min(max(cell_y, 0), GRID_SIZE_Y - 1)

        # Add the boid index to the grid cell if there is space
        count = grid_counts[cell_x, cell_y]

        if count > MAX_BOIDS_PER_CELL: continue

        grid[cell_x, cell_y, count] = i
        grid_counts[cell_x, cell_y] = count + 1


@ti.func
def toroidal_wrap(offset):
    """
    Apply toroidal wrapping to the given offset vector. This function ensures that the simulation space is periodic,
    meaning that boids wrapping around the edges of the screen reappear on the opposite side. This is achieved by
    adjusting the offset coordinates to be within the bounds of the screen dimensions.

    Args:
        offset (ti.Vector): The offset vector to wrap.

    Returns:
        ti.Vector: The wrapped offset vector.
    """
    if offset.x > WIDTH / 2: offset.x -= WIDTH
    if offset.x < -WIDTH / 2: offset.x += WIDTH
    if offset.y > HEIGHT / 2: offset.y -= HEIGHT
    if offset.y < -HEIGHT / 2: offset.y += HEIGHT
    return offset


@ti.kernel
def update():
    """    
    Update the positions and velocities of all boids based on the flocking behavior rules. This kernel computes the
    alignment, cohesion, and separation forces for each boid by examining its neighbors within the relevant grid cells.
    The spatial partitioning grid is used to efficiently find neighboring boids, significantly speeding up the simulation.

    Mathematical Explanation:
    - Alignment: Steer towards the average velocity of nearby boids. This is achieved by summing the velocities of neighbors
      and normalizing the result to compute the desired velocity direction.
    - Cohesion: Steer towards the center of mass of nearby boids. The center of mass is calculated as the average position
      of neighbors, and the desired direction is the vector pointing from the boid's position to this center.
    - Separation: Steer away from nearby boids that are too close. This is done by summing the normalized vectors pointing
      away from each close neighbor, scaled inversely by their distance to emphasize avoidance of closer boids.

    The forces are combined with respective weights to compute the acceleration for each boid. The velocity is updated
    based on this acceleration, and the speed is clamped to a maximum value to ensure stability.
    """
    for i in range(NUM_BOIDS):
        pos_i = positions[i]
        vel_i = velocities[i]

        # Initialize force vectors and counters
        align = ti.Vector([0.0, 0.0]) 
        cohere = ti.Vector([0.0, 0.0])  
        separate = ti.Vector([0.0, 0.0]) 
        count = 0
        count_sep = 0

        # Find the grid cell of the current boid
        cell_x = int(pos_i.x // CELL_SIZE)
        cell_y = int(pos_i.y // CELL_SIZE)
        cell_x = min(max(cell_x, 0), GRID_SIZE_X - 1)
        cell_y = min(max(cell_y, 0), GRID_SIZE_Y - 1)

        # Check neighbors in adjacent cells (3x3 neighborhood)
        # `ti.static` is a Taichi construct that allows certain operations, such as loops, to be
        # unrolled at compile time. This improves performance by avoiding runtime overhead and
        # enabling optimizations. For example, in the 3x3 neighborhood loop, `ti.static` ensures
        # that the loop bounds are fixed and known at compile time, allowing the compiler to
        # generate efficient code.
        for dx in ti.static(range(-2, 3)):
            for dy in ti.static(range(-2, 3)):
                nx = cell_x + dx
                ny = cell_y + dy
                if 0 <= nx < GRID_SIZE_X and 0 <= ny < GRID_SIZE_Y:
                    for idx in range(grid_counts[nx, ny]):
                        j = grid[nx, ny, idx]

                        if i == j: continue

                        pos_j = positions[j]
                        offset = toroidal_wrap(pos_j - pos_i)  # Handle toroidal wrapping

                        dist = offset.norm()  # Euclidean distance between boid i and boid j

                        if dist < NEIGHBOR_RADIUS:
                            # Alignment: Sum the velocities of neighbors
                            align += velocities[j]

                            # Cohesion: Sum the positions of neighbors to compute the center of mass
                            cohere += pos_j
                            count += 1

                            if dist < SEPARATION_RADIUS:
                                # Separation: Steer away from close neighbors
                                if dist > 1e-5:
                                    separate -= offset.normalized() / dist  # Weighted by inverse distance
                                else:
                                     # Random perturbation if boids are to close. 
                                     # Had to add this because without it there was weird behaviour where boids were able to colapse into each other. 
                                    separate += ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) 
                                count_sep += 1

        acc = ti.Vector([0.0, 0.0])  # Acceleration vector of boid

        if count > 0:
            # Alignment: Compute the desired velocity direction based on the average velocity of neighbors
            align_norm = align.norm()
            if align_norm > 1e-5:
                align = (align / align_norm) * MAX_SPEED - vel_i  # Desired velocity change
                acc += alignment_weight * align * 0.01  # Scale alignment force

            # Cohesion: Compute the desired direction towards the center of mass of neighbors
            center = cohere / count  # Center of mass
            desired = center - pos_i  # Vector pointing to the center
            desired = toroidal_wrap(desired)  # Handle toroidal wrapping

            desired_norm = desired.norm()
            if desired_norm > 1e-5:
                desired = (desired / desired_norm) * MAX_SPEED - vel_i  # Desired velocity change
                acc += cohesion_weight * desired * 0.005  # Scale cohesion force

        if count_sep > 0:
            # Separation: Compute the desired direction away from close neighbors
            sep_norm = separate.norm()
            if sep_norm > 1e-5:
                separate = (separate / sep_norm) * MAX_SPEED - vel_i  # Desired velocity change
                acc += separation_weight * separate * 0.15  # Scale separation force

        # Update velocity with the computed acceleration
        vel_i += acc

        # Clamp the speed to the maximum allowed value
        speed = vel_i.norm()
        if speed > MAX_SPEED:
            vel_i = (vel_i / speed) * MAX_SPEED

        # Update the velocity and position of the boid
        velocities[i] = vel_i
        positions[i] = (pos_i + vel_i) % ti.Vector([WIDTH, HEIGHT])  # Apply toroidal wrapping to positions


initialize()

gui = ti.GUI("Taichi Boids with Spatial Partitioning", res=(WIDTH, HEIGHT))
print(f"Running on: {ti.lang.impl.current_cfg().arch}")

while gui.running:
    build_grid()
    update()
    pos_np = positions.to_numpy()
    pos_np[:, 0] /= WIDTH
    pos_np[:, 1] /= HEIGHT
    gui.circles(pos_np, radius=1.5, color=0xFFFFFF)
    gui.show()
