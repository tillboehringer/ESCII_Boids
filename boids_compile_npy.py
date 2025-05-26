import numpy as np
import cv2

# Load saved positions: shape = (num_frames, num_boids, 2)
positions = np.load("boids_positions.npy")

num_frames, num_boids, _ = positions.shape
WIDTH, HEIGHT = 1600, 1000

# Setup OpenCV video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' codec for mp4 files
fps = 30
out = cv2.VideoWriter('boids_simulation.mp4', fourcc, fps, (WIDTH, HEIGHT))

for frame_idx in range(num_frames):
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)  # black background
    
    # Get boid positions for this frame
    pos = positions[frame_idx]

    # Draw boids as white circles (convert float coords to int pixels)
    for (x, y) in pos:
        px = int(x)
        py = int(y)
        cv2.circle(frame, (px, py), radius=2, color=(255, 255, 255), thickness=-1)
    
    out.write(frame)

    if (frame_idx + 1) % 10 == 0 or frame_idx == num_frames - 1:
        print(f"Writing video frame {frame_idx + 1}/{num_frames}")

out.release()
print("MP4 video saved as 'boids_simulation.mp4'")
