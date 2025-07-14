import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm

class MotionPrimitives:
    def __init__(self, u_max=2.0, r=2, tau=0.5, n=2):
        """
        Initialize motion primitives generator
        
        Args:
            u_max: Maximum control input magnitude
            r: Discretization parameter (gives (2r+1)^3 primitives)
            tau: Primitive duration
            n: Order (2 for double integrator)
        """
        self.u_max = u_max
        self.r = r
        self.tau = tau
        self.n = n
        
        # Generate discretized control inputs
        self.generate_control_inputs()
        
        # System matrices for double integrator (n=2)
        self.A = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        
        self.B = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
    
    def generate_control_inputs(self):
        """Generate discretized control inputs U_D"""
        # Create discretized values for each axis
        axis_values = []
        for i in range(2 * self.r + 1):
            if i == 0:
                axis_values.append(-self.u_max)
            elif i == 2 * self.r:
                axis_values.append(self.u_max)
            else:
                axis_values.append(-self.u_max + (2 * self.u_max * i) / (2 * self.r))
        
        # Generate all combinations for 3D control
        self.U_D = []
        for ux in axis_values:
            for uy in axis_values:
                for uz in axis_values:
                    self.U_D.append(np.array([ux, uy, uz]))
        
        print(f"Generated {len(self.U_D)} control primitives")
    
    def generate_primitive(self, x0, u_d):
        """
        Generate a motion primitive from initial state x0 with control u_d
        
        Args:
            x0: Initial state [px, py, pz, vx, vy, vz]
            u_d: Discretized control input [ux, uy, uz]
            
        Returns:
            trajectory: Array of states over time
            times: Time points
        """
        # Time discretization for visualization
        dt = 0.01
        t_steps = int(self.tau / dt) + 1
        times = np.linspace(0, self.tau, t_steps)
        
        # Initialize trajectory
        trajectory = np.zeros((t_steps, 6))
        trajectory[0] = x0
        
        # Integration using matrix exponential (exact solution)
        for i, t in enumerate(times[1:], 1):
            # x(t) = e^(At) * x(0) + integral_0^t e^(A(t-tau)) * B * u(tau) dtau
            # For constant u, this becomes: e^(At) * x(0) + A^(-1) * (e^(At) - I) * B * u
            
            eAt = expm(self.A * t)
            
            # For double integrator, we can compute the integral analytically
            # The solution is: x(t) = x0 + v0*t + 0.5*u*t^2, v(t) = v0 + u*t
            pos_part = x0[:3] + x0[3:6] * t + 0.5 * u_d * t**2
            vel_part = x0[3:6] + u_d * t
            
            trajectory[i] = np.concatenate([pos_part, vel_part])
        
        return trajectory, times
    
    def visualize_primitives_2d(self, x0=None, max_primitives=50):
        """
        Visualize motion primitives in 2D (similar to Figure 2)
        
        Args:
            x0: Initial state [px, py, pz, vx, vy, vz]
            max_primitives: Maximum number of primitives to show
        """
        if x0 is None:
            x0 = np.array([0, 0, 0, 0, 0, 0])  # Start from origin with zero velocity
        
        plt.figure(figsize=(12, 8))
        
        # Plot a subset of primitives for clarity
        primitive_indices = np.random.choice(len(self.U_D), 
                                           min(max_primitives, len(self.U_D)), 
                                           replace=False)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(primitive_indices)))
        
        for i, idx in enumerate(primitive_indices):
            u_d = self.U_D[idx]
            trajectory, times = self.generate_primitive(x0, u_d)
            
            # Plot XY projection
            plt.plot(trajectory[:, 0], trajectory[:, 1], 
                    color=colors[i], alpha=0.7, linewidth=1.5)
            
            # Mark the end point
            plt.plot(trajectory[-1, 0], trajectory[-1, 1], 
                    'o', color=colors[i], markersize=4)
        
        # Mark starting point
        plt.plot(x0[0], x0[1], 'rs', markersize=10, label='Start')
        
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title(f'Motion Primitives Visualization (2D)\n{len(primitive_indices)} primitives, τ={self.tau}s, u_max={self.u_max}')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.legend()
        
        # Add text with parameters
        textstr = f'Parameters:\nr = {self.r} (gives {len(self.U_D)} primitives)\nτ = {self.tau}s\nu_max = {self.u_max}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_primitives_3d(self, x0=None, max_primitives=30):
        """
        Visualize motion primitives in 3D
        
        Args:
            x0: Initial state [px, py, pz, vx, vy, vz]
            max_primitives: Maximum number of primitives to show
        """
        if x0 is None:
            x0 = np.array([0, 0, 0, 0, 0, 0])
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot a subset of primitives for clarity
        primitive_indices = np.random.choice(len(self.U_D), 
                                           min(max_primitives, len(self.U_D)), 
                                           replace=False)
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(primitive_indices)))
        
        for i, idx in enumerate(primitive_indices):
            u_d = self.U_D[idx]
            trajectory, times = self.generate_primitive(x0, u_d)
            
            # Plot 3D trajectory
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                   color=colors[i], alpha=0.7, linewidth=2)
            
            # Mark the end point
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
                      color=colors[i], s=30)
        
        # Mark starting point
        ax.scatter(x0[0], x0[1], x0[2], color='red', s=100, marker='s', label='Start')
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title(f'Motion Primitives Visualization (3D)\n{len(primitive_indices)} primitives')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def analyze_primitive_properties(self):
        """Analyze properties of the generated primitives"""
        x0 = np.array([0, 0, 0, 0, 0, 0])  # Start from origin
        
        end_positions = []
        end_velocities = []
        costs = []
        
        for u_d in self.U_D:
            trajectory, times = self.generate_primitive(x0, u_d)
            end_positions.append(trajectory[-1, :3])
            end_velocities.append(trajectory[-1, 3:6])
            
            # Calculate cost according to Equation 4: J(T) = ∫||u(t)||²dt + ρT
            rho = 1.0  # Time penalty
            control_cost = np.linalg.norm(u_d)**2 * self.tau
            time_cost = rho * self.tau
            costs.append(control_cost + time_cost)
        
        end_positions = np.array(end_positions)
        end_velocities = np.array(end_velocities)
        costs = np.array(costs)
        
        print(f"\nPrimitive Analysis:")
        print(f"Number of primitives: {len(self.U_D)}")
        print(f"Position range: X[{end_positions[:, 0].min():.2f}, {end_positions[:, 0].max():.2f}], "
              f"Y[{end_positions[:, 1].min():.2f}, {end_positions[:, 1].max():.2f}], "
              f"Z[{end_positions[:, 2].min():.2f}, {end_positions[:, 2].max():.2f}]")
        print(f"Velocity range: VX[{end_velocities[:, 0].min():.2f}, {end_velocities[:, 0].max():.2f}], "
              f"VY[{end_velocities[:, 1].min():.2f}, {end_velocities[:, 1].max():.2f}], "
              f"VZ[{end_velocities[:, 2].min():.2f}, {end_velocities[:, 2].max():.2f}]")
        print(f"Cost range: [{costs.min():.2f}, {costs.max():.2f}]")

# Example usage and demonstration
if __name__ == "__main__":
    # Create motion primitives generator with paper's parameters
    mp = MotionPrimitives(u_max=2.0, r=2, tau=0.5, n=2)
    
    # Analyze primitive properties
    mp.analyze_primitive_properties()
    
    # Visualize primitives in 2D (similar to Figure 2)
    print("\nGenerating 2D visualization...")
    mp.visualize_primitives_2d(max_primitives=40)
    
    # Visualize primitives in 3D
    print("Generating 3D visualization...")
    mp.visualize_primitives_3d(max_primitives=25)
    
    # Example with different initial conditions
    print("Generating visualization with initial velocity...")
    x0_with_velocity = np.array([0, 0, 0, 1, 1, 0])  # Start with some initial velocity
    mp.visualize_primitives_2d(x0=x0_with_velocity, max_primitives=100)