import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from ergodic_control import utilities
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point
# Import path
from matplotlib.path import Path

class Robot:
    def __init__(self, x, y, param, dt=1):
        self.position = np.array([x, y])
        self.dt = dt
        self.x_hist = np.empty((0, 2))
        self.neighbors = []
        self.param = param
        self.bbox = [param.x_inf, param.y_inf, param.x_sup, param.y_sup]
        self.range = param.range
        self.x_grid = param.x_grid
        self.y_grid = param.y_grid
        self.grid = param.grid
        self.combo_density = np.zeros_like(self.grid)
        self.mu = np.zeros_like(self.grid)
        self.std = np.zeros_like(self.grid)

    def update(self, u):
        self.position[0] += u[0] * self.dt
        self.position[1] += u[1] * self.dt
        self.x_hist = np.vstack((self.x_hist, self.position))

    def compute_diagram(self, vertices):
        intersection_points = np.empty((0, 2))
        # Intersect the Voronoi region with the closed polygon centered at the robot position
        polygon = Polygon(vertices)
        circle = Point(self.position).buffer(self.range * 0.5)
        intersection = polygon.intersection(circle)
        if intersection.is_empty:
            intersection_points = np.empty((0, 2))
        elif intersection.geom_type == 'Polygon':
            intersection_points = np.array(list(intersection.exterior.coords))
        elif intersection.geom_type == 'MultiPolygon':
            intersection_points = np.empty((0, 2))
            for poly in intersection:
                intersection_points = np.append(intersection_points, list(poly.exterior.coords), axis=0)

        self.diagram = intersection_points


    def compute_centroid(self, t):      
        vertices = self.diagram
        dA = param.dx**2

        p = Path(vertices)
        bool_val = p.contains_points(self.grid)
        mu = self.mu[bool_val.reshape(self.x_grid.shape[0], self.y_grid.shape[0])]
        std = self.std[bool_val.reshape(self.x_grid.shape[0], self.y_grid.shape[0])]

        weight = std + np.tanh(0.1 * t) * mu * 10
        weight = np.exp(weight) - 1

        A = np.sum(weight) * dA
        Cx = np.sum(self.grid[:, 0][bool_val] * weight) * dA / A
        Cy = np.sum(self.grid[:, 1][bool_val] * weight) * dA / A
    
        return np.array([Cx, Cy], dtype=np.float64)

def full_voronoi_diagram(robots, bbox):
    robot_positions = np.array([robot.position for robot in robots])

    vertices = []

    points_left = np.copy(robot_positions)
    points_left[:, 0] = 2 * bbox[0] - points_left[:, 0]
    points_right = np.copy(robot_positions)
    points_right[:, 0] = 2 * bbox[2] - points_right[:, 0]
    points_down = np.copy(robot_positions)
    points_down[:, 1] = 2 * bbox[1] - points_down[:, 1]
    points_up = np.copy(robot_positions)
    points_up[:, 1] = 2 * bbox[3] - points_up[:, 1]
    points = np.vstack((robot_positions, points_left, points_right, points_down, points_up))

    # Voronoi diagram
    vor = Voronoi(points)
    vor.filtered_points = robot_positions
    vor.filtered_regions = [vor.regions[i] for i in vor.point_region[:len(robot_positions)]]
    for region in vor.filtered_regions:
        vertices.append(vor.vertices[region + [region[0]], :])

    return vertices

np.random.seed(87)

param = lambda: None
param.x_inf, param.y_inf = 0, 0
param.x_sup, param.y_sup = 50, 50
param.dx = 1

param.nbAgents = 8
param.range = 16
PERIOD = 5000

# Create the grid
param.x_grid = np.arange(param.x_inf, param.x_sup + param.dx, param.dx)
param.y_grid = np.arange(param.y_inf, param.y_sup + param.dx, param.dx)
# param.x_grid, param.y_grid = np.meshgrid(np.arange(param.x_inf, param.x_sup), np.arange(param.y_inf, param.y_sup), indexing='ij')

x_mesh, y_mesh = np.meshgrid(param.x_grid, param.y_grid)
param.grid = np.c_[x_mesh.ravel(), y_mesh.ravel()]
width = height = len(param.x_grid)

mean = [40, 40]
cov = [[25, 0], [0, 25]]

field = utilities.gauss_pdf(param.grid, mean, cov).reshape(x_mesh.shape)

robots = np.empty(param.nbAgents, dtype=object)
for r in np.arange(param.nbAgents):
    x1, x2 = np.random.uniform(0+1, param.x_sup / 2), np.random.uniform(0+1, param.y_sup / 2)
    robots[r] = Robot(x1, x2, param)

noise = 0
kernel = C(1.0) * RBF(1.0) # + WhiteKernel(1e-5)
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=False)
std_pred_test = np.zeros_like(field)
preSamplesN = 500
preSamples = np.random.randint(0, param.x_sup, (preSamplesN, 2))
preSamples = np.hstack((preSamples, field.ravel()[np.ravel_multi_index((preSamples[:, 0], preSamples[:, 1]), field.shape)].reshape(-1, 1)))
preSamples = np.unique(preSamples, axis=0, return_index=False)
gpr.fit(preSamples[:, :2], preSamples[:, 2])

for robot in robots:
    robot.dataset = np.empty((0, 3))
    robot.neighbors = []
    robot.field = field
    robot.coverage_density = np.zeros_like(field)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.contourf(x_mesh, y_mesh, field, cmap='viridis')
ax.scatter([robot.position[0] for robot in robots], [robot.position[1] for robot in robots], c='red')

robot_block = utilities.agent_block(2, 1e-8, agent_radius=param.range)
kernel_size = robot_block.shape[0]
ergodic_metrics = np.zeros((param.nbAgents, PERIOD))

for t in np.arange(0, PERIOD):
    print("Time step: ", t)
    vertices = full_voronoi_diagram(robots, [param.x_inf, param.y_inf, param.x_sup, param.y_sup])

    for i, robot in enumerate(robots):
        # Resets
        robot.neighbors = []
        subset = np.empty((0, 3))
        samples = np.empty((0, 3))

        # Set the neighbors
        for other_robot in robots:
            if robot != other_robot:
                if np.linalg.norm([robot.position[0] - other_robot.position[0], robot.position[1] - other_robot.position[1]]) < param.range:
                    robot.neighbors.append(other_robot)
                    robot.datset = np.vstack((robot.dataset, other_robot.dataset))
        
        # Update the voronoi diagram
        robot.compute_diagram(vertices[i])

        # Include the grid cells inside the robot range
        x_min, x_max = robot.diagram[:, 0].astype(int).min(), robot.diagram[:, 0].astype(int).max()
        y_min, y_max = robot.diagram[:, 1].astype(int).min(), robot.diagram[:, 1].astype(int).max()

        local_x_grid = np.arange(x_min, x_max, param.dx)
        local_y_grid = np.arange(y_min, y_max, param.dx)
        local_x_mesh, local_y_mesh = np.meshgrid(local_x_grid, local_y_grid)
        local_grid = np.column_stack((local_x_mesh.ravel(), local_y_mesh.ravel()))

        path = Path(robot.diagram)
        mask = path.contains_points(local_grid)
        local_grid = local_grid[mask]
        local_field = utilities.gauss_pdf(local_grid, mean, cov)

        samples = np.column_stack((local_grid, local_field + noise * np.random.randn(len(local_field))))

        if t > 0:
            std_test = robot.std_pred_test[samples[:, 0].astype(int), samples[:, 1].astype(int)]
            samples = samples[np.where(std_test > 0.1)[0]]

            if len(samples) != 0:
                subset = np.unique(np.vstack((subset, utilities.max_pooling(samples, 5, divisor_range=(2, 5)))), axis=0, return_index=False)
                robot.dataset = np.vstack((robot.dataset, subset))
                # Clear duplicates
                _, idx = np.unique(robot.dataset[:, :2], axis=0, return_index=True)
                robot.dataset = robot.dataset[idx]

                robot.combo_density, robot.mu, robot.std, robot.std_pred_test = utilities.compute_combo(robot.dataset, 
                                                            param.grid, 
                                                            np.zeros_like(field), 
                                                            gpr.kernel_)
                robot.std[robot.std < 0.1] = 0
        else:
            subset = np.unique(np.vstack((subset, utilities.max_pooling(samples, 5, divisor_range=(2, 5)))), axis=0, return_index=False)
            robot.dataset = np.vstack((robot.dataset, subset))
            # Clear duplicates
            _, idx = np.unique(robot.dataset[:, :2], axis=0, return_index=True)
            robot.dataset = robot.dataset[idx]
            robot.combo_density, robot.mu, robot.std, robot.std_pred_test = utilities.compute_combo(robot.dataset, 
                                                          param.grid, 
                                                          np.zeros_like(field), 
                                                          gpr.kernel_)
            robot.std[robot.std < 0.1] = 0

        # Compute the centroid
        centroid = robot.compute_centroid(t)

        adjusted_position = np.array([robot.position[0], robot.position[1]])
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

        goal_density = robot.std + np.tanh(0.1 * t) * robot.mu * 10
        goal_density = np.exp(goal_density) - 1
        diff = utilities.normalize_mat(goal_density) - utilities.normalize_mat(robot.coverage_density)
        source = np.maximum(diff, 0) ** 2 # Eq. 13 - Source term
        _em_diff = np.sum(source/np.linalg.norm(source)) * param.dx * param.dx
        ergodic_metrics[i, t] = _em_diff

        robot.update(1 * (centroid - robot.position))
        print(f"Robot {i} dataset: {robot.dataset.shape}")

# Save the paths
plt.close("all")
filepath = os.path.join(os.path.dirname(__file__), "sim3")
for i, robot in enumerate(robots):
    np.save(os.path.join(filepath, f"robot_{i}.npy"), robot.x_hist)

# Save the ergodic metrics
np.save(os.path.join(filepath, "ergodic_metrics.npy"), ergodic_metrics)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.contourf(x_mesh, y_mesh, field, cmap='viridis')
ax.scatter([robot.position[0] for robot in robots], [robot.position[1] for robot in robots], 
            marker="o", 
            facecolors='none', 
            edgecolors='red',
            s=100)
plt.show()

# Plot the ergodic metrics
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
for i in np.arange(param.nbAgents):
    ax.plot(ergodic_metrics[i], label=f"Robot {i}")
ax.legend()
plt.show()


robot = robots[0]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(121)
ax.set_aspect('equal')
ax.contourf(x_mesh, y_mesh, robot.mu.reshape(x_mesh.shape), cmap='Reds')
ax.scatter(robot.dataset[:, 0], robot.dataset[:, 1], c='red')
ax.plot(robot.x_hist[:, 0], robot.x_hist[:, 1], c='black')
ax.scatter(robot.x_hist[0, 0], robot.x_hist[0, 1], c='green')
ax.scatter(robot.x_hist[-1, 0], robot.x_hist[-1, 1], c='blue')

ax = fig.add_subplot(122)
ax.set_aspect('equal')
ax.contourf(x_mesh, y_mesh, robot.std.reshape(x_mesh.shape), cmap='Blues')
plt.show()

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.set_aspect('equal')
ax.contourf(x_mesh, y_mesh, robot.combo_density.reshape(x_mesh.shape), cmap='viridis')
ax.scatter(robot.dataset[:, 0], robot.dataset[:, 1], c='red')
ax.plot(robot.x_hist[:, 0], robot.x_hist[:, 1], c='black')
ax.scatter(robot.x_hist[0, 0], robot.x_hist[0, 1], c='green')
ax.scatter(robot.x_hist[-1, 0], robot.x_hist[-1, 1], c='blue')
plt.show()