import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

np.random.seed(42)  # For reproducibility

# -------------------------------
# 1. Define the combined kernel
# -------------------------------
# Constant kernel * RBF(space) * RBF(time) + WhiteKernel(noise)
kernel = (
    C(1.0, constant_value_bounds=(1e-3, 1e3))
    * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))  # space
    * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))  # time
    + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e1))
)

# Initialize the Gaussian Process Regressor
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-5, normalize_y=False)

# -------------------------------
# 2. Generate synthetic training data
# -------------------------------
# Synthetic spatial domain
N_data = 100
X_train = np.random.uniform(0, 200, N_data).reshape(-1, 1)
X_train = np.sort(X_train, axis=0)  # Sort spatial points

# Synthetic temporal domain
T_train = np.random.uniform(0, 20, N_data).reshape(-1, 1)
# Sort temporal points
T_train = np.sort(T_train, axis=0)

# Synthetic observations: sinusoidal pattern + noise
def f(x):
    return np.sin(0.05 * x) + 0.05 * np.random.randn(*x.shape)

y_train = f(X_train).ravel()

# -------------------------------
# 4. Fit the Gaussian Process model
# -------------------------------
gp.fit(np.hstack((X_train, T_train)), y_train)

# -------------------------------
# 5. Predict over a test spatial domain (at a fixed time)
# -------------------------------
X_test = np.linspace(0, 200, 200).reshape(-1, 1)
T_test = np.ones_like(X_test) * 20  # Fixed time point for testing
X_test_input = np.hstack((X_test, T_test))

y_pred, sigma = gp.predict(X_test_input, return_std=True)

# -------------------------------
# 6. Plot predictions with confidence intervals
# -------------------------------

# Print optimized kernel
print("Optimized kernel:", gp.kernel_)

plt.figure(figsize=(10, 5))
plt.plot(X_test, y_pred, color='blue', label='Predicted Mean')
plt.fill_between(
    X_test.ravel(),
    y_pred - 1.96 * sigma,
    y_pred + 1.96 * sigma,
    color='lightblue',
    alpha=0.5,
    label='95% Confidence Interval'
)
# Change the color of the training data points based on the time
plt.scatter(X_train, y_train, cmap='viridis', c=T_train.ravel(), s=50, edgecolor='k', label='Training Data')
plt.title('Gaussian Process Regression Predictions')
plt.xlabel('Spatial Coordinate')
plt.ylabel('Observation')
plt.legend()
plt.grid()
plt.show()


