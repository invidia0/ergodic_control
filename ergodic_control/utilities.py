import numpy as np
from matplotlib.collections import LineCollection
from sklearn.mixture import GaussianMixture
from shapely.geometry import Point, Polygon
from matplotlib.path import Path
import warnings
import time
from skimage.measure import find_contours
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.vectorized import contains
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.linalg import cholesky, solve_triangular
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import sklearn.gaussian_process.kernels as kernels
from scipy.signal import convolve2d
from scipy.ndimage import convolve

def timeit(method):
    """
    A decorator to measure the execution time of a function.
    """
    # Suppress 
    return method
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(f"{method.__name__} executed in: {te - ts:.6f} sec")
        return result
    return timed

def update_heat(heat, source, local_cooling, map, param):
    # Define the Laplacian kernel

    # Convolve heat with the Laplacian kernel
    # laplacian = convolve2d(heat, param.laplacian_kernel, mode='same', boundary='symm')
    laplacian = convolve(heat, param.laplacian_kernel, mode='reflect')

    # Update heat equation
    new_temperature = heat + param.dt * (
                        param.alpha * laplacian + 
                        param.source_strength * source -
                        param.beta * heat
                        )
    new_temperature[map != 0] = heat[map != 0]  # Keep occupied cells unchanged

    return new_temperature

# def update_heat(heat, source, local_cooling, map, param):
#     # Compute the laplacian for all grid cells
#     laplacian = (
#         np.roll(heat, 1, axis=0)  # Shift up
#         + np.roll(heat, -1, axis=0)  # Shift down
#         + np.roll(heat, 1, axis=1)  # Shift left
#         + np.roll(heat, -1, axis=1)  # Shift right
#         - 4 * heat
#     ) / param.dx**2

#     new_temperature = heat + param.dt * (param.alpha * laplacian + param.source_strength * source)
#     new_temperature[map != 0] = heat[map != 0]  # Keep occupied cells unchanged

#     return new_temperature

def gauss_pdf(points, mean, covariance):
  # Calculate the multivariate Gaussian probability
  exponent = -0.5 * np.sum((points - mean) @ np.linalg.inv(covariance) * (points - mean), axis=1)
  coefficient = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covariance))
  prob = coefficient * np.exp(exponent)

  return prob

def gmm_eval(points, means, covariances, weights):
  prob = 0.0
  s = len(means)
  for i in range(s):
    prob += weights[i] * gauss_pdf(points, means[i], covariances[i])

  return prob

def compute_gmm(param):
    """
    Compute the PDF of the GMM.
    """
    # How to consider the weights?
    samples = np.array([
        np.random.multivariate_normal(np.array(param._mu)[i, :], np.array(param._sigmas)[i, :, :], param.nbParticles)
        for i in range(param.nbGaussian)
    ]).reshape(param.nbGaussian * param.nbParticles, param.nbVarX)

    gmm = GaussianMixture(n_components=param.nbGaussian, covariance_type='full').fit(samples)

    return gmm.means_, gmm.covariances_, gmm.weights_

def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)


def normalize_mat(mat):
    """
    Normalize a matrix to sum to 1.
    """
    return mat / np.add(np.sum(mat), 1e-10)

def min_max_normalize(mat):
    """
    Normalize a matrix to [0, 1].
    """
    return (mat - np.min(mat)) / (np.max(mat) - np.min(mat) + 1e-10)

def hadamard_matrix(n: int) -> np.ndarray:
    """
    Constructs a Hadamard matrix of size n.

    Args:
        n (int): The size of the Hadamard matrix.

    Returns:
        np.ndarray: A Hadamard matrix of size n.
    """
    # Base case: A Hadamard matrix of size 1 is just [[1]].
    if n == 1:
        return np.array([[1]])

    # Recursively construct a Hadamard matrix of size n/2.
    half_size = n // 2
    h_half = hadamard_matrix(half_size)

    # Combine the four sub-matrices to form a Hadamard matrix of size n.
    h = np.empty((n, n), dtype=int)
    h[:half_size,:half_size] = h_half
    h[half_size:,:half_size] = h_half
    h[:half_size:,half_size:] = h_half
    h[half_size:,half_size:] = -h_half

    return h

def discrete_gmm(param):
    """
    Mixture models for the analysis, edition, and synthesis of continuous time series
    Sylvain Calinon

    https://calinon.ch/papers/Calinon_MMchapter2019.pdf
    """
    # Discretize given GMM using Fourier basis functions
    rg = np.arange(0, param.nbFct, dtype=float)
    KX = np.zeros((param.nbVarX, param.nbFct, param.nbFct))
    KX[0, :, :], KX[1, :, :] = np.meshgrid(rg, rg)

    # Explicit description of w_hat by exploiting the Fourier transform
    # properties of Gaussians (optimized version by exploiting symmetries)
    Lambda = np.array(KX[0, :].flatten() ** 2 + KX[1, :].flatten() ** 2 + 1).T ** (-(param.nbVar + 1) / 2)

    op = hadamard_matrix(2 ** (param.nbVarX - 1))
    op = np.array(op)
    # check the reshaping dimension !!!
    kk = KX.reshape(param.nbVarX, param.nbFct**2) * param.omega

    # Compute fourier basis function weights w_hat for the target distribution given by GMM
    w_hat = np.zeros(param.nbFct**param.nbVarX)
    for j in range(param.nbGaussian):
        for n in range(op.shape[1]):
            MuTmp = np.diag(op[:, n]) @ param.Means[:, j]
            SigmaTmp = np.diag(op[:, n]) @ param.Covs[:, :, j] @ np.diag(op[:, n]).T
            cos_term = np.cos(kk.T @ MuTmp)
            exp_term = np.exp(np.diag(-0.5 * kk.T @ SigmaTmp @ kk))
            # Eq.(22) where D=1
            w_hat = w_hat + param.Weights[j] * cos_term * exp_term
    w_hat = w_hat / (param.L**param.nbVarX) / (op.shape[1])

    # Fourier basis functions (for a discretized map)
    xm1d = np.linspace(param.xlim[0], param.xlim[1], param.nbRes)  # Spatial range
    xm = np.zeros((param.nbGaussian, param.nbRes, param.nbRes))
    xm[0, :, :], xm[1, :, :] = np.meshgrid(xm1d, xm1d)
    # Mind the flatten() !!!
    ang1 = (
        KX[0, :, :].flatten().T[:, np.newaxis]
        @ xm[0, :, :].flatten()[:, np.newaxis].T
        * param.omega
    )
    ang2 = (
        KX[1, :, :].flatten().T[:, np.newaxis]
        @ xm[1, :, :].flatten()[:, np.newaxis].T
        * param.omega
    )
    phim = np.cos(ang1) * np.cos(ang2) * 2 ** (param.nbVarX)
    # Some weird +1, -1 due to 0 index !!!
    xx, yy = np.meshgrid(np.arange(1, param.nbFct + 1), np.arange(1, param.nbFct + 1))
    hk = np.concatenate(([1], 2 * np.ones(param.nbFct)))
    HK = hk[xx.flatten() - 1] * hk[yy.flatten() - 1]
    phim = phim * np.tile(HK, (param.nbRes**param.nbVarX, 1)).T

    # Desired spatial distribution
    g = w_hat.T @ phim
    return g, w_hat, phim, xx, yy, rg, Lambda

def discrete_gmm_original(param):
   
    ks_dim1, ks_dim2 = np.meshgrid(
        np.arange(param.nbFct), np.arange(param.nbFct)
    )
    ks = np.array([ks_dim1.ravel(), ks_dim2.ravel()]).T

    L_list = np.array([param.xlim[1], param.xlim[1]])
    grids_x, grids_y = np.meshgrid(
        np.linspace(0, L_list[0], 100),
        np.linspace(0, L_list[1], 100)
    )
    grids = np.array([grids_x.ravel(), grids_y.ravel()]).T

    dx = L_list[1] / param.nbRes
    dy = L_list[1] / param.nbRes

    coefficients = np.zeros(ks.shape[0])  # number of coefficients matches the number of index vectors
    pdf_vals = gmm_eval(grids, param.Means, param.Covs, param.Weights)  # this can computed ahead of the time

    for i, k_vec in enumerate(ks):
        # step 1: evaluate the fourier basis function over all the grid cells
        fk_vals = np.prod(np.cos(np.pi * k_vec / L_list * grids), axis=1)  # we use NumPy's broadcasting feature to simplify computation
        hk = np.sqrt(np.sum(np.square(fk_vals)) * dx * dy)  # normalization term
        fk_vals /= hk

        # step 3: approximate the integral through the Riemann sum for the coefficient
        phik = np.sum(fk_vals * pdf_vals) * dx * dy 
        coefficients[i] = phik

    pdf_recon = np.zeros(grids.shape[0])
    for i, (phik, k_vec) in enumerate(zip(coefficients, ks)):
        fk_vals = np.prod(np.cos(np.pi * k_vec / L_list * grids), axis=1)
        hk = np.sqrt(np.sum(np.square(fk_vals)) * dx * dy)
        fk_vals /= hk
        
        pdf_recon += phik * fk_vals

    return pdf_recon

def rbf(mean, x, eps):
    """
    Radial basis function w/ Gaussian Kernel
    """
    d = mean - x  # radial distance
    l2_norm_squared = np.dot(d, d)

    return np.exp(-eps * l2_norm_squared)

def agent_block(nbVarX, min_val, agent_radius):
    """
    A matrix representing the shape of an agent (e.g, RBF with Gaussian kernel). 
    min_val is the upper bound on the minimum value of the agent block.
    
    RBF(d) = exp(-(eps * d)^2)
    
    eps = 1 / agent_radius 
    is the shape parameter that can be interpreted as the inverse of the radius
    
    Example usage:
    If min_val = 0.1 and agent_radius = 2, the function creates a matrix where:
        - RBF values are computed up to where the minimum value is 0.1.
        - The block size reflects the radius of influence based on the Gaussian decay rate.
    """

    eps = 1.0 / agent_radius  # shape parameter of the RBF (supposed to be squared)
    l2_sqrd = (
        -np.log(min_val) / eps
    )  # squared maximum distance from the center of the agent block
    l2_sqrd_single = (
        l2_sqrd / nbVarX
    )  # maximum squared distance on a single axis since sum of all axes equal to l2_sqrd
    l2_single = np.sqrt(l2_sqrd_single)  # maximum distance on a single axis
    # round to the nearest larger integer
    if l2_single.is_integer(): 
        l2_upper = int(l2_single)
    else:
        l2_upper = int(l2_single) + 1
    # agent block is symmetric about the center
    num_rows = l2_upper * 2 + 1
    num_cols = num_rows
    block = np.zeros((num_rows, num_cols))
    center = np.array([num_rows // 2, num_cols // 2])
    for i in range(num_rows):
        for j in range(num_cols):
            block[i, j] = rbf(center, np.array([j, i]), eps)
    # we hope this value is close to zero 
    print(f"Minimum element of the block: {np.min(block)}" +
          " values smaller than this assumed as zero")
    return block

def offset(mat, i, j):
    """
    offset a 2D matrix by i, j
    """
    rows, cols = mat.shape
    rows = rows - 2
    cols = cols - 2
    return mat[1 + i : 1 + i + rows, 1 + j : 1 + j + cols]

# def clamp_kernel_1d(x, low_lim, high_lim, kernel_size):
#     """
#     A function to calculate the start and end indices
#     of the kernel around the agent that is inside the grid
#     i.e. clamp the kernel by the grid boundaries
#     """
#     start_kernel = low_lim
#     start_grid = x - (kernel_size // 2)
#     num_kernel = kernel_size
#     # bound the agent to be inside the grid
#     if x <= -(kernel_size // 2):
#         x = -(kernel_size // 2) + 1
#     elif x >= high_lim + (kernel_size // 2):
#         x = high_lim + (kernel_size // 2) - 1

#     # if agent kernel around the agent is outside the grid,
#     # clamp the kernel by the grid boundaries
#     if start_grid < low_lim:
#         start_kernel = kernel_size // 2 - x - 1
#         num_kernel = kernel_size - start_kernel - 1
#         start_grid = low_lim
#     elif start_grid + kernel_size >= high_lim:
#         num_kernel -= x - (high_lim - num_kernel // 2 - 1)
#     if num_kernel > low_lim:
#         grid_indices = slice(start_grid, start_grid + num_kernel)

#     return grid_indices, start_kernel, num_kernel

def border_interpolate(x, length, border_type):
    """
    Helper function to interpolate border values based on the border type
    (gives the functionality of cv2.borderInterpolate function)
    """
    if border_type == "reflect101":
        if x < 0:
            return -x
        elif x >= length:
            return 2 * length - x - 2
    return x

def bilinear_interpolation(grid, pos):
    """
    Linear interpolating function on a 2-D grid
    """
    x, y = pos.astype(int)
    # find the nearest integers by minding the borders
    x0 = border_interpolate(x, grid.shape[1], "reflect101")
    x1 = border_interpolate(x + 1, grid.shape[1], "reflect101")
    y0 = border_interpolate(y, grid.shape[0], "reflect101")
    y1 = border_interpolate(y + 1, grid.shape[0], "reflect101")
    # Distance from lower integers
    xd = pos[0] - x0
    yd = pos[1] - y0
    # Interpolate on x-axis
    c01 = grid[y0, x0] * (1 - xd) + grid[y0, x1] * xd
    c11 = grid[y1, x0] * (1 - xd) + grid[y1, x1] * xd
    # Interpolate on y-axis
    c = c01 * (1 - yd) + c11 * yd
    return c

def clamp_kernel_1d(x, low_lim, high_lim, kernel_size, occupancy_grid=None, axis=None):
    """
    Calculate the start and end indices of the kernel around the agent
    that is inside the grid while avoiding walls.
    """
    start_kernel = low_lim
    start_grid = x - (kernel_size // 2)
    num_kernel = kernel_size

    # Bound the agent to be inside the grid
    if x <= -(kernel_size // 2):
        x = -(kernel_size // 2) + 1
    elif x >= high_lim + (kernel_size // 2):
        x = high_lim + (kernel_size // 2) - 1

    # Clamp the kernel to grid boundaries
    if start_grid < low_lim:
        start_kernel = kernel_size // 2 - x - 1
        num_kernel = kernel_size - start_kernel - 1
        start_grid = low_lim
    elif start_grid + kernel_size >= high_lim:
        num_kernel -= x - (high_lim - num_kernel // 2 - 1)
    if num_kernel > low_lim:
        grid_indices = slice(start_grid, start_grid + num_kernel)

    # Avoid walls if occupancy grid is provided
    if occupancy_grid is not None and axis is not None:
        for i in range(grid_indices.start, grid_indices.stop):
            if axis == "row" and occupancy_grid[i, x] == 1:  # Avoid walls in rows
                start_grid = max(i - 1, low_lim)
                num_kernel -= 1
            elif axis == "col" and occupancy_grid[x, i] == 1:  # Avoid walls in cols
                start_grid = max(i - 1, low_lim)
                num_kernel -= 1

    return grid_indices, start_kernel, num_kernel

def calculate_gradient_map(param, agent, gradient_x, gradient_y, occupancy_grid):
    """
    Calculate movement direction of the agent considering heading,
    the gradient of the field, and wall avoidance.
    """
    adjusted_position = agent.x
    x, y = adjusted_position.astype(int)
    heading_vector = np.array([np.cos(agent.theta), np.sin(agent.theta)])
    gradient = np.zeros(2)

    if 0 <= x < param.width and 0 <= y < param.height:
        gradient[0] = bilinear_interpolation(gradient_x, adjusted_position)
        gradient[1] = bilinear_interpolation(gradient_y, adjusted_position)

    kernel_radius = param.kernel_size // 2
    wall_boundary_gradient = 1e-4
    sigma = kernel_radius

    wall_effect = np.zeros(2)
    #obstacles = np.empty((0, 2))
    for dx in range(-kernel_radius, kernel_radius + 1):
        for dy in range(-kernel_radius, kernel_radius + 1):
            next_x = x + dx
            next_y = y + dy

            if 0 <= next_x < param.width and 0 <= next_y < param.height:
                if occupancy_grid[next_x, next_y] == 1:
                    #obstacles = np.vstack([obstacles, [next_x, next_y]])
                    distance = np.sqrt(dx**2 + dy**2)
                    if distance > 0:
                        influence = wall_boundary_gradient * np.exp(-distance**2 / (2 * sigma**2))
                        direction = np.array([-dx / distance, -dy / distance])  # Away from the wall
                        wall_effect += influence * direction

    # wall_effect_norm = np.linalg.norm(wall_effect)
    # if wall_effect_norm > 0:
    #     wall_effect /= wall_effect_norm

    # Project wall effect onto the heading
    parallel_effect = np.dot(wall_effect, heading_vector) * heading_vector
    perpendicular_effect = wall_effect - parallel_effect

    # Scale perpendicular effect to prevent sharp turns
    perpendicular_scaling = 0.5  # Adjust sensitivity
    wall_effect = (1 - perpendicular_scaling) * parallel_effect + perpendicular_scaling * perpendicular_effect

    # Combine interpolated gradient and wall effect
    gradient += wall_effect

    # Normalize the resulting gradient to prevent erratic movements
    norm = np.linalg.norm(gradient)
    if norm > 0:
        gradient /= norm

    return gradient#, obstacles

# def calculate_gradient_map(param, agent, gradient_x, gradient_y, occupancy_grid):
#     """
#     Calculate movement direction of the agent by considering the gradient
#     of the temperature field near the agent, avoiding walls.
#     """
#     # Find agent position on the grid as integer indices
#     adjusted_position = agent.x
#     col, row = adjusted_position.astype(int)  # x corresponds to col, y to row

#     gradient = np.zeros(2)

#     # Check if agent is inside the grid
#     if 0 <= row < param.width and 0 <= col < param.height:
#         # Interpolate the gradient for the agent's position
#         gradient[0] = bilinear_interpolation(gradient_x, adjusted_position)
#         gradient[1] = bilinear_interpolation(gradient_y, adjusted_position)

#     # Always check for walls within the kernel size and push the agent away
#     kernel_radius = param.kernel_size // 2
#     wall_boundary_gradient = 10  # Scaling factor for wall influence
#     sigma = kernel_radius  # Spread of the Gaussian influence

#     wall_effect_x = 0
#     wall_effect_y = 0

#     for dr in range(-kernel_radius, kernel_radius + 1):
#         for dc in range(-kernel_radius, kernel_radius + 1):
#             neighbor_row = row + dr
#             neighbor_col = col + dc

#             if 0 <= neighbor_row < param.width and 0 <= neighbor_col < param.height:
#                 if occupancy_grid[neighbor_row, neighbor_col] == 1:  # Wall detected
#                     # Calculate the Euclidean distance
#                     distance = np.sqrt(dr**2 + dc**2)
#                     if distance > 0:
#                         # Gaussian influence based on distance
#                         influence = wall_boundary_gradient * np.exp(-distance**2 / (2 * sigma**2))
#                         # Normalized direction away from the wall
#                         direction_x = -dc / distance
#                         direction_y = -dr / distance
#                         wall_effect_x += influence * direction_x
#                         wall_effect_y += influence * direction_y

#     # Combine the interpolated gradient and wall avoidance effect
#     gradient[0] += wall_effect_x
#     gradient[1] += wall_effect_y

#     # Normalize the resulting gradient to prevent erratic movements
#     norm = np.linalg.norm(gradient)
#     if norm > 0:
#         gradient /= norm

#     return gradient

def draw_fov(pos, theta, fov, fov_depth):
    """
    This function returns a list of points that make up the field of view.
    
    Parameters:
    pos: list or array-like, position of the agent [x, y]
    theta: float, heading angle of the agent in radians
    fov: float, field of view in degrees (e.g., 90)
    fov_depth: float, depth of the field of view
    
    Returns:
    np.array: A numpy array of points representing the field of view in 2D space.
    """
    # Convert FOV from degrees to radians
    fov_rad = np.radians(fov)
    
    # Calculate the angles for the left and right FOV boundaries
    left_angle = theta + fov_rad / 2
    right_angle = theta - fov_rad / 2
    
    # Calculate the end points of the FOV lines
    left_point = [
        pos[0] + fov_depth * np.cos(left_angle),
        pos[1] + fov_depth * np.sin(left_angle)
    ]
    right_point = [
        pos[0] + fov_depth * np.cos(right_angle),
        pos[1] + fov_depth * np.sin(right_angle)
    ]
    
    # The FOV is represented by a triangle: [position, left_point, right_point]
    fov_points = [pos, left_point, right_point]
    
    return np.array(fov_points)


def draw_fov_arc(pos, theta, fov, fov_depth, num_points=50):
    """
    This function returns a list of points along the arc that describes the field of view.
    
    Parameters:
    pos: list or array-like, position of the agent [x, y]
    theta: float, heading angle of the agent in radians
    fov: float, field of view in degrees (e.g., 90)
    fov_depth: float, depth of the field of view
    num_points: int, number of points along the arc to describe the FOV
    
    Returns:
    np.array: A numpy array of points representing the arc of the field of view in 2D space.
    """
    # Convert FOV from degrees to radians
    fov_rad = np.radians(fov)
    
    # Calculate the angles for the left and right FOV boundaries
    left_angle = theta + fov_rad / 2
    right_angle = theta - fov_rad / 2
    
    # Generate points along the arc between left_angle and right_angle
    angles = np.linspace(left_angle, right_angle, num_points)
    
    # Calculate the x and y coordinates for the points along the arc
    arc_points = np.array([
        [pos[0] + fov_depth * np.cos(angle), pos[1] + fov_depth * np.sin(angle)]
        for angle in angles
    ])
    arc_points = np.vstack([pos, arc_points, pos])
    return arc_points

def simple_gaussian_fov_block(points, fov_arc, fov_depth, bounds):
    fov_arc_clamped = np.clip(fov_arc, [bounds[0], bounds[1]], [bounds[2], bounds[3]])

    poly = Path(fov_arc_clamped)
    fov_points = np.ceil([point for point in points if poly.contains_point(point)]).astype(int)
    fov_center = np.mean(fov_arc_clamped, axis=0)

    prob = np.exp(-np.linalg.norm(fov_points - fov_center, axis=1) / (fov_depth / 3))

    prob = (prob - np.min(prob)) / (np.max(prob) - np.min(prob))

    return fov_points, prob

@timeit
def init_fov(fov_deg, fov_depth):
    """
    This function returns a list of points that make up the field of view.
    
    Parameters:
    pos: list or array-like, position of the agent [x, y] or an arbitrary initial position.
    theta: float, heading angle of the agent in radians
    fov: float, field of view in degrees (e.g., 90)
    fov_depth: float, depth of the field of view
    
    Returns:
    np.array: A numpy array of points representing the field of view in 2D space.
    """
    # Generate a triangle fov with the tip of the triangle in (0, 0)
    fov_rad = np.radians(fov_deg)

    # Calculate the angles for the left and right FOV boundaries
    left_angle = fov_rad / 2
    right_angle = -fov_rad / 2

    # Calculate the end points of the FOV lines
    left_point = [
        fov_depth * np.cos(left_angle),
        fov_depth * np.sin(left_angle)
    ]
    right_point = [
        fov_depth * np.cos(right_angle),
        fov_depth * np.sin(right_angle)
    ]

    return np.array([[0, 0], left_point, right_point])

@timeit
def insidepolygon(polygon, grid_step=1):
    """
    Returns a list of points that lie within a non-convex polygon defined by its vertices.
    
    Parameters:
    - polygon: List or array of (x, y) coordinates representing the polygon vertices.
    - grid_step: Spacing between grid points to sample inside the polygon.
    
    Returns:
    - points_inside: Numpy array of points [[x1, y1], [x2, y2], ...] inside the polygon.
    """
    polygon_shapely = Polygon(polygon)
    
    # Get bounding box directly from the polygon
    min_x, min_y, max_x, max_y = polygon_shapely.bounds
    
    # Generate grid points within the bounding box
    x_range = np.arange(min_x, max_x, grid_step)
    y_range = np.arange(min_y, max_y, grid_step)
    X, Y = np.meshgrid(x_range, y_range)
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    
    # Use vectorized contains to check points
    mask = contains(polygon_shapely, grid_points[:, 0], grid_points[:, 1])

    return grid_points[mask]

def rotate_and_translate(fov_array, pos, theta):
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return np.dot(fov_array, rot_matrix.T) + pos

@timeit
def fov_coverage_block(points, fov_array, fov_depth):
    """
    This function calculates the probability of each grid point within the field of view.
    This is a simple function with a Gaussian distribution centered at the middle of the FOV.

    Parameters:
    points: np.array, a numpy array of grid points in the environment (discrete coordinates).
    fov_array: np.array, points describing the arc of the FOV boundary.
    fov_depth: float, the maximum depth of the FOV.
    """
    # Find the center of the FOV
    fov_center = np.mean(fov_array, axis=0)

    # Define a simple Gaussian distribution around the center of the FOV
    prob = np.exp(-np.linalg.norm(points - fov_center, axis=1) / (fov_depth / 3))

    # Normalize the probabilities such that at boundaries the probability is zero
    prob = (prob - np.min(prob)) / (np.max(prob) - np.min(prob))

    return prob

@timeit
def clip_polygon(subject_polygon, clip_polygon):
    """ Sutherland-Hodgman Polygon Clipping Algorithm """
    def inside(p, cp):
        return (cp[1, 0] - cp[0, 0]) * (p[1] - cp[0, 1]) > (cp[1, 1] - cp[0, 1]) * (p[0] - cp[0, 0])

    def intersection(cp, s, e):
        dc = cp[0] - cp[1]
        dp = s - e
        n1 = np.cross(cp[0], cp[1])
        n2 = np.cross(s, e)
        n3 = 1.0 / np.cross(dc, dp)
        return (n1 * dp - n2 * dc) * n3

    output_list = subject_polygon
    cp1 = clip_polygon[-1]
    for cp2 in clip_polygon:
        input_list = output_list
        output_list = []
        s = input_list[-1]
        for e in input_list:
            if inside(e, np.array([cp1, cp2])):
                if not inside(s, np.array([cp1, cp2])):
                    output_list.append(intersection(np.array([cp1, cp2]), s, e))
                output_list.append(e)
            elif inside(s, np.array([cp1, cp2])):
                output_list.append(intersection(np.array([cp1, cp2]), s, e))
            s = e
        cp1 = cp2
    return np.array(output_list)


def translate_grid(relative_grid, center):
    return relative_grid + center

@timeit
def relative_move_fov(fov_array, old_pos, old_head, new_pos, new_head):
    """
    Moves and rotates the FOV array from the old agent position and heading to the new agent position and heading.

    Parameters:
    - fov_array: np.array, shape (3, 2), points of the FOV boundary (already aligned to the old position and heading).
    - old_pos: np.array, shape (2,), the old position of the agent.
    - old_head: float, the old orientation (in radians).
    - new_pos: np.array, shape (2,), the new position of the agent.
    - new_head: float, the new orientation (in radians).

    Returns:
    - transformed_fov: np.array, shape (3, 2), the new FOV points.
    """
    # Step 1: Translation vector to move from old position to new position
    trans_v = new_pos - old_pos

    # Step 2: Compute the relative rotation angle
    rel_rot = new_head - old_head

    # Rotation matrix for the relative rotation
    rot_matrix = np.array([
        [np.cos(rel_rot), -np.sin(rel_rot)],
        [np.sin(rel_rot), np.cos(rel_rot)]
    ])

    # Translate FOV to the new position
    trans_fov = fov_array + trans_v

    # Rotate FOV around the new agent position
    return np.dot(trans_fov - new_pos, rot_matrix.T) + new_pos

def get_occupied_polygon(map_array):
    contours = find_contours(map_array, level=0.5)
    if contours:
        return np.array(contours[0])
    else:
        return np.array([])

@timeit
def clip_polygon_no_convex(agent_pos, fov_polygon, occupied_polygon):
    """
    Clips the field-of-view (fov_polygon) by the occupied area (occupied_polygon),
    and returns the resulting polygon closest to the agent position.
    
    Parameters:
    - agent_pos: A tuple or list with the agent's position (x0, y0).
    - fov_polygon: List or array of points representing the field-of-view polygon.
    - occupied_polygon: List or array of points representing the occupied polygon.
    
    Returns:
    - clipped_polygon: A numpy array of coordinates of the clipped polygon.
    """
    agent_pos = Point(agent_pos)
    
    # Convert polygons to Shapely objects
    fov_poly = Polygon(fov_polygon)
    occ_map = Polygon(occupied_polygon)
    
    # Perform clipping
    diff = fov_poly.difference(occ_map)
    
    # Handle multiple intersections (if it's a MultiPolygon)
    if isinstance(diff, MultiPolygon):
        diff = min(diff.geoms, key=lambda poly: poly.distance(agent_pos))
    
    # Check if the resulting clipped polygon is not empty
    if not diff.is_empty:
        return np.array(diff.exterior.coords)
    else:
        return np.array([])

def generate_gmm_on_map(map, free_cells, n_gaussians, n_particles, dim, random_state=None):
    """
    Generates a Gaussian Mixture Model (GMM) on a map where walls block Gaussian influence.

    Parameters:
        map (np.ndarray): Map array where 0 represents free cells and 1 represents walls.
        free_cells (np.ndarray): Array of free cells (coordinates), shape (n_free_cells, dim).
        n_gaussians (int): Number of Gaussian peaks to generate.
        n_particles (int): Number of particles to sample for each Gaussian.
        dim (int): Dimensionality of the map (e.g., 2 for 2D).
        random_state (int, optional): Seed for reproducibility.

    Returns:
        gmm (GaussianMixture): Trained Gaussian Mixture Model.
        zi_masked (np.ndarray): Values of the GMM evaluated on the map, with walls masked as NaN.
    """
    if random_state:
        np.random.seed(random_state)

    # Randomly select means from the free cells
    means_idx = np.random.choice(len(free_cells), size=n_gaussians, replace=False)
    means = free_cells[means_idx]

    # Generate random covariances
    covariances = []
    for _ in range(n_gaussians):
        cov = np.diag(np.random.uniform(20, 800, size=dim))  # Restrict covariance values
        covariances.append(cov)

    # Generate samples
    samples = []
    for mean, cov in zip(means, covariances):
        for _ in range(n_particles):
            while True:
                sample = np.random.multivariate_normal(mean, cov)
                # Check bounds and free cells
                if (0 <= sample[0] < map.shape[0]) and (0 <= sample[1] < map.shape[1]) and map[int(sample[0]), int(sample[1])] == 0:
                    samples.append(sample)
                    break
    samples = np.array(samples)

    # Fit the GMM
    gmm = GaussianMixture(n_components=n_gaussians, covariance_type='full', random_state=random_state)
    gmm.fit(samples)

    # Evaluate the GMM only at free cells
    gmm_density = gmm.score_samples(free_cells)
    density_at_free_cells = np.exp(gmm_density)

    # Create a map-shaped density array
    zi = np.full(map.shape, np.nan)  # Initialize with NaN
    zi[free_cells[:, 0], free_cells[:, 1]] = density_at_free_cells  # Assign densities to free cells

    return gmm, zi

@timeit
def max_pooling(data, downsampling_factor, divisor_range=(2, 5)):
    """
    Performs max pooling on a dataset of samples (x, y, f(x, y)).
    
    Parameters:
    - data (np.ndarray): Input dataset of shape (N, 3), where each row is (x, y, f(x, y)).
    - downsampling_factor (int): Factor to reduce the number of samples.
    - divisor_range (tuple): Range of divisors to consider for approximation (inclusive).
    
    Returns:
    - pooled_data (np.ndarray): Downsampled dataset of shape (M, 3), where M is determined by the adjusted downsampling factor.
    """
    if len(data) == 0 or len(data) == 1:
        return data

    n_samples = len(data)

    # Adjust the downsampling factor if it's not feasible
    while n_samples % downsampling_factor != 0:
        # Find a divisor in the given range
        for factor in range(divisor_range[1], divisor_range[0] - 1, -1):
            if n_samples % factor == 0:
                downsampling_factor = factor
                break
        else:
            # Remove one random sample to make the downsampling factor valid
            data = np.delete(data, np.random.randint(0, n_samples), axis=0)
            n_samples -= 1
            
    chunk_size = n_samples // downsampling_factor
    pooled_data = []

    # Iterate through chunks of data
    for i in range(0, n_samples, chunk_size):
        chunk = data[i:i + chunk_size]
        if len(chunk) > 0:  # Handle edge cases where the last chunk might be incomplete
            max_idx = np.argmax(chunk[:, 2])
            pooled_data.append(chunk[max_idx])

    return np.array(pooled_data)


# def rbf_white_kernel(X1: np.ndarray, 
#                      X2: np.ndarray,
#                      lengthscale: float=1.0,
#                      sigma_f: float=1.0,
#                      sigma_n: float=1e-8) -> np.ndarray:
#     """
#     Exponentiated Quadratic Kernel with optimizations for large matrices.
    
#     Args:
#         X1: Array of m points (m x d).
#         X2: Array of n points (n x d).

#     Returns:
#         (m x n) matrix.
#     """
#     # Precompute squared norms
#     X1_sqnorm = np.sum(X1**2, axis=1)[:, np.newaxis]  # m x 1
#     X2_sqnorm = np.sum(X2**2, axis=1)  # n

#     # Efficient computation of pairwise squared Euclidean distance
#     sqdist = X1_sqnorm + X2_sqnorm - 2 * np.dot(X1, X2.T)

#     # Apply kernel function (Exponentiated Quadratic)
#     SQE = np.exp(-0.5 * sqdist / lengthscale**2)

#     # Factor out the constant for the kernel
#     C = sigma_f**2

#     # Return the kernel matrix with the scaling factor
#     return C * SQE  # + np.eye(X1.shape[0]) * sigma_n**2 (optional regularization)

def rbf_white_kernel(X1: np.ndarray, 
                     X2: np.ndarray = None,
                     lengthscale: float = 1.0,
                     constant: float = 1.0,
                     sigma_n: float = 1e-8,
                     diag=False) -> np.ndarray:
    """
    Exponentiated Quadratic Kernel with a constant term for large matrices.

    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d). If None, computes k(X1, X1).
        lengthscale: Length scale of the kernel.
        sigma_f: Signal variance (controls amplitude of the RBF kernel).
        sigma_n: Noise variance (added to diagonal for k(X1, X1)).
        sigma_c: Constant variance term (added to all kernel entries).

    Returns:
        Kernel matrix of shape (m x n).
    """
    if diag == False:
        if X2 is None:
            # Use pdist for k(X1, X1)
            scaled_X1 = X1 / lengthscale
            dists = pdist(scaled_X1, metric="sqeuclidean")
            K = np.exp(-0.5 * squareform(dists))
            # Add noise variance to the diagonal
            np.fill_diagonal(K, np.diag(K) + sigma_n**2)
        else:
            # Use cdist for k(X1, X2)
            scaled_X1 = X1 / lengthscale
            scaled_X2 = X2 / lengthscale
            dists = cdist(scaled_X1, scaled_X2, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)
            # Add noise variance to the diagonal
            if X1 is X2:
                np.fill_diagonal(K, np.diag(K) + sigma_n**2)

        # Add constant term
        K += constant**2
    else:
        # Compute the diagonal of the kernel matrix
        K = np.ones(X1.shape[0]) * constant ** 2
        K += np.exp(-0.5 * np.sum((X1 / lengthscale) ** 2, axis=1))
    return K

def gp_predict(X_train: np.ndarray, 
                y_train: np.ndarray, 
                X_test: np.ndarray, 
                kernel: kernels.Kernel):
    """
    Optimized GP prediction function with optional covariance or standard deviation.

    Args:
        X_train: Training data (n_train x d).
        y_train: Training targets (n_train,).
        X_test: Test data (n_test x d).
        lengthscale: Kernel lengthscale.
        constant: Kernel constant term.
        sigma_n: Noise variance.

    Returns:
        mu: Predicted mean (n_test,).
        std: Predicted standard deviation (n_test,).
    """
    # Compute the training kernel matrix
    K = kernel(X_train) + np.eye(X_train.shape[0]) * 1e-10
    L = cholesky(K, lower=True, check_finite=False)

    # Compute alpha = (L.T @ L)^-1 @ y_train_normalized
    alpha = solve_triangular(L, y_train, lower=True)
    alpha = solve_triangular(L.T, alpha, lower=False)

    # Kernel between training and test points
    K_s = kernel(X_train, X_test)
    # Mean prediction
    mu = K_s.T @ alpha

    V = solve_triangular(L, K_s, lower=True, check_finite=False)

    # Compute variance: diag(K(X_test, X_test)) - sum(V^2)
    K_ss = kernel.diag(X_test).copy()

    # Use einsum to compute the diagonal of V^T @ V efficiently
    var = K_ss - np.einsum("ij,ij->j", V, V)

    # Numerical stability for variance
    var = np.maximum(var, 0)

    # Compute standard deviation
    std = np.sqrt(var)

    return mu, std