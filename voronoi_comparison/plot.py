import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from ergodic_control import utilities

class Robot:
    def __init__(self, x_hist, dt=1):
        # self.x = x
        # self.y = y
        self.dt = dt
        self.x_hist = x_hist
        self.coverage_density = np.zeros_like(field)

    def update(self, ut):
        self.x = self.x + self.dt * ut[0]
        self.y = self.y + self.dt * ut[1]
        self.x_hist = np.vstack((self.x_hist, [self.x, self.y]))

robots_hists = []
nbRobot = 8

# filepath = os.path.join(os.path.dirname(__file__), "sim1/")
filepath = os.path.join(os.path.dirname(__file__))

for i in range(nbRobot):
    robots_hists.append(np.load(filepath + "/robot_" + str(i) + ".npy"))

param = lambda: None
param.x_inf, param.y_inf = 0, 0
param.x_sup, param.y_sup = 50, 50
param.dx = 1

param.nbAgents = 8
CAMERA_BOX = 10
param.range = 10
PERIOD = 3000

# Create the grid
param.x_grid = np.arange(param.x_inf, param.x_sup + param.dx, param.dx)
param.y_grid = np.arange(param.y_inf, param.y_sup + param.dx, param.dx)
# param.x_grid, param.y_grid = np.meshgrid(np.arange(param.x_inf, param.x_sup), np.arange(param.y_inf, param.y_sup), indexing='ij')

x_mesh, y_mesh = np.meshgrid(param.x_grid, param.y_grid)
param.grid = np.c_[x_mesh.ravel(), y_mesh.ravel()]

mean = [40, 40]
cov = [[25, 0], [0, 25]]

height = len(param.y_grid)
width = len(param.x_grid)

field = utilities.gauss_pdf(param.grid, mean, cov).reshape(x_mesh.shape)

# Initialize robots
robots = []
for i in range(nbRobot):
    robots.append(Robot(robots_hists[i], dt=1))

for robot in robots:
    robot.coverage_density = np.zeros_like(field)

# Generate an RBF at the robot position
robot_block = utilities.agent_block(2, 1e-8, agent_radius=10)
kernel_size = robot_block.shape[0]
ergodic_metrics = np.zeros((PERIOD, nbRobot))
coverage = np.zeros_like(field)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
for i in range(PERIOD):
    # ax.clear()
    # ax.set_aspect('equal')
    # ax.set_xlim(param.x_inf, param.x_sup)
    # ax.set_ylim(param.y_inf, param.y_sup)
    for j, robot in enumerate(robots):
        adjusted_position = np.array([robot.x_hist[i, 0], robot.x_hist[i, 1]])
        y, x = adjusted_position.astype(int)

        x_indices, x_start_kernel, num_kernel_dx = utilities.clamp_kernel_1d(
            x, 0, width, kernel_size
        )
        y_indices, y_start_kernel, num_kernel_dy = utilities.clamp_kernel_1d(
            y, 0, height, kernel_size
        )
        robot.coverage_density[x_indices, y_indices] += utilities.normalize_mat(robot_block[
            x_start_kernel : x_start_kernel + num_kernel_dx,
            y_start_kernel : y_start_kernel + num_kernel_dy,
        ])

        diff = utilities.normalize_mat(field) - utilities.normalize_mat(robot.coverage_density)
        source = np.maximum(diff, 0) ** 2 # Eq. 13 - Source term
        _em_diff = np.sum(source/np.linalg.norm(source)) * param.dx * param.dx
        ergodic_metrics[i, j] = _em_diff

        # if j == 0:
        #     ax.contourf(x_mesh, y_mesh, robots[j].coverage_density, cmap="Greens")
        #     ax.plot(robots[j].x_hist[:i, 0], robots[j].x_hist[:i, 1], label='robot' + str(j))
        #     ax.scatter(robots[j].x_hist[i, 0], robots[j].x_hist[i, 1], s=100, facecolors='none', edgecolors='r', linewidth=2)
    # plt.pause(0.01)

plt.show()




# Plot the ergodic metric
fig, ax = plt.subplots()
for i in range(nbRobot):
    ax.plot(ergodic_metrics[:, i], label='robot' + str(i))
plt.show()

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(param.x_inf, param.x_sup)
ax.set_ylim(param.y_inf, param.y_sup)
ax.contourf(x_mesh, y_mesh, field, cmap='viridis')
for i in range(nbRobot):
    ax.plot(robots_hists[i][:, 0], robots_hists[i][:, 1], label='robot' + str(i))
    # Scatter a thick red circle at the last position
    ax.scatter(robots_hists[i][-1, 0], robots_hists[i][-1, 1], s=100, facecolors='none', edgecolors='r', linewidth=2)
plt.show()