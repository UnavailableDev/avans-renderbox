import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define constants
NUM_PARTICLES = 1024
TIME_STEP = 0.01
GRAVITY = 9.81
RESTITUTION = 0.8 # Coefficient for particle collsion
RADIUS = 0.05

# OpenCL kernel
with open('particle_kernel.cl', 'r') as f:
    kernel_code = f.read();

# Initialize data
positions = np.random.rand(NUM_PARTICLES, 4).astype(np.float32)  # Random positions
# velocities = np.random.rand(NUM_PARTICLES, 4).astype(np.float32)  # Random velocities
velocities = np.zeros((NUM_PARTICLES, 4), dtype=np.float32)       # Initially zero velocities

# OpenCL setup
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Create buffers
mf = cl.mem_flags
positions_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=positions)
velocities_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=velocities)

# Compile kernel
program = cl.Program(context, kernel_code).build()

# Set up the plot
fig, ax = plt.subplots()
scat = ax.scatter(positions[:, 0], positions[:, 1])
# quiver = ax.quiver(positions[:, 0], positions[:, 1], velocities[:, 0], velocities[:, 1])

ax.set_xlim(0, 2)
ax.set_ylim(0, 4)

# Animation update function
def update(frame):
    program.update_particles(queue, (NUM_PARTICLES,), None, positions_buf, velocities_buf, np.float32(TIME_STEP), np.float32(GRAVITY), np.float32(RESTITUTION), np.float32(RADIUS), np.int32(NUM_PARTICLES))

    cl.enqueue_copy(queue, positions, positions_buf).wait()
    cl.enqueue_copy(queue, velocities, velocities_buf).wait()

    # Compute the color based on speed
    # speed = np.linalg.norm(velocities[:, :2], axis=1)
    
    speed = np.linalg.norm(velocities[:, :3], axis=1)
    max_speed = np.max(speed)
    colors = speed / max_speed if max_speed > 0 else speed   # (velocities[:, 0], velocities[:, 1], speed)

    scat.set_offsets(positions[:, :2])
    scat.set_array(colors)
    return scat,

    # quiver.set_offsets(positions[:, :2])
    # quiver.set_UVC(velocities[:, 0], velocities[:, 1], speed)
    # 
    # return quiver,

# Run the animation
ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)
plt.show()

print("Simulation completed.")


