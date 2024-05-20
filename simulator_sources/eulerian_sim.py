import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define constants
# NUM_PARTICLES = 1024
TIME_STEP = 0.01
GRAVITY = -9.81
RESTITUTION = 0.8 # Coefficient for particle collsion
RADIUS = 0.05
OVERRELAXATION = 1.0 # value should be 1.0 < x < 2.0

WIDTH = 10
HEIGHT = 10

# OpenCL kernel
with open('navier_stokes.cl', 'r') as f:
    kernel_code = f.read();

# Initialize data
num_cells = WIDTH * HEIGHT
num_u = (WIDTH + 1) * HEIGHT
num_v = (HEIGHT + 1) * WIDTH
# velocities = np.random.rand(NUM_PARTICLES, 4).astype(np.float32)  # Random velocities
vel_u = np.random.rand((num_u)).astype(np.float32)       # Initially zero velocities
vel_v = np.random.rand((num_v)).astype(np.float32) * 5      # Initially zero velocities
# vel_u = np.zeros((num_u), dtype=np.float32)       # Initially zero velocities
# vel_v = np.zeros((num_v), dtype=np.float32)       # Initially zero velocities
fluids = np.ones((num_cells), dtype=np.float32)
pressures = np.zeros((num_cells), dtype=np.float32)

# Set boundaries
fluids[0:WIDTH] = 0.0
fluids[0:num_cells:WIDTH] = -1.0
fluids[WIDTH-1:num_cells:WIDTH] = 0.0
fluids[num_cells-WIDTH:] = 0.0

# Load object into windtunnel


# OpenCL setup
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Create buffers
mf = cl.mem_flags
vel_u_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=vel_u)
vel_v_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=vel_v)
fluids_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=fluids)
pressures_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=pressures)

# Compile kernel
program = cl.Program(context, kernel_code).build()

# Set up the plot
fig, ax = plt.subplots()
# scat = ax.scatter(positions[:, 0], positions[:, 1])
X, Y = np.meshgrid(np.arange(0,WIDTH), np.arange(0,HEIGHT))
quiver = ax.quiver(X, Y, vel_u[:num_cells], vel_v[:num_cells])

ax.set_xlim(-0.5, WIDTH)
ax.set_ylim(-0.5, HEIGHT)
# quiver.set_offsets()



# Animation update function
def update(frame):
    # TODO Fix the next 2 functions
    program.update_velocity(queue, (num_cells,), None, vel_u_buf, vel_v_buf, fluids_buf, np.float32(TIME_STEP), np.float32([0.0, GRAVITY]))
    program.update_divergence(queue, (num_cells,), None, vel_u_buf, vel_v_buf, np.float32(TIME_STEP), np.float32(WIDTH), np.float32(HEIGHT), fluids_buf, np.float32(OVERRELAXATION))
    # program.update_advection(queue, (num_cells,), None, vel_u_buf, vel_v_buf, np.float32(TIME_STEP), np.float32(WIDTH), np.float32(HEIGHT))
    
    # upd_velocity_knl = program.update_velocity
    # upd_divergence_knl = program.update_divergence
    # upd_advection_knl = program.update_advection

    # upd_velocity_knl.set_args(vel_u_buf, vel_v_buf, np.float32(TIME_STEP), np.float32([0.0, GRAVITY]))
    # upd_divergence_knl.set_args(vel_u_buf, vel_v_buf, np.float32(TIME_STEP), np.float32(WIDTH), np.float32(HEIGHT), fluids_buf, np.float32(OVERRELAXATION))
    # upd_advection_knl.set_args(vel_u_buf, vel_v_buf, np.float32(TIME_STEP), np.float32(WIDTH), np.float32(HEIGHT))

    # cl.enqueue_nd_range_kernel(queue, upd_velocity_knl, (WIDTH, HEIGHT), None)
    # cl.enqueue_nd_range_kernel(queue, upd_divergence_knl, (WIDTH, HEIGHT), None)
    # cl.enqueue_nd_range_kernel(queue, upd_advection_knl, (WIDTH, HEIGHT), None)

    cl.enqueue_copy(queue, vel_u, vel_u_buf).wait()
    cl.enqueue_copy(queue, vel_v, vel_v_buf).wait()

    # Compute the color based on speed
    # speed = np.linalg.norm(velocities[:, :2], axis=1)
    
    # TODO Fix colorations
    speed = np.linalg.norm(vel_v)
    max_speed = np.max(speed)
    colors = speed / max_speed if max_speed > 0 else speed   # (velocities[:, 0], velocities[:, 1], speed)

    # scat.set_offsets(positions[:, :2])
    # scat.set_array(colors)
    # return scat,

    # TODO Fix this
    # quiver.set_offsets(positions[:, :2])
    quiver.set_UVC(vel_u[:num_cells], vel_v[:num_cells], fluids)

    # Print Debugging
    print(f"Vel_v max: {np.max(vel_v)}, avg: {np.average(vel_v)}")
    
    return quiver,

# Run the animation
ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)
plt.show()

print("Simulation completed.")


