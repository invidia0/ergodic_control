import numpy as np

class FirstOrderAgent:
    """
    A point mass agent with first order dynamics.
    """
    def __init__(
        self,
        x, # initial position
        dt=1, # time step
        max_ut=1, # maximum velocity
    ):
        self.x = np.array(x)  # position
        self.nbVarX = len(x)

        self.dt = dt  # time step

        self.max_ut = max_ut

        self.x_hist = np.empty((0, self.nbVarX))

    def update(self, ut):
        """
        set the velocity of the agent to clamped gradient
        compute the position at t+1 based on clamped velocity
        ut: control input
        """
        if np.linalg.norm(ut) > self.max_ut:
            ut = self.max_ut * ut / np.linalg.norm(ut)

        self.x = self.x + self.dt * ut
        self.x_hist = np.vstack((self.x_hist, self.x))


class DoubleIntegratorAgent:
    def __init__(
        self,
        x,  # initial position [x, y]
        theta=0,  # initial heading
        max_dx=1,  # maximum velocity
        max_ddx=2.5,  # maximum acceleration
        max_dtheta=1.0,  # maximum angular velocity
        max_ddtheta=2.5,  # maximum angular acceleration
        dt=1,  # time step
        id=0,  # agent id
    ):
        self.x = np.array(x, dtype=np.float64)  # Position [x, y]
        self.v = np.array([0.0, 0.0], dtype=np.float64)  # Velocity [vx, vy]
        self.theta = theta  # Heading angle
        self.max_dx = max_dx
        self.max_ddx = max_ddx
        self.max_dtheta = max_dtheta
        self.max_ddtheta = max_ddtheta
        self.omega = 0.0  # Angular velocity
        self.dt = dt
        self.id = id
        self.x_hist = np.empty((0, 3))  # History of states [x, y, theta]

    def update(self, u):
        """
        Update the state of the agent given the control input.

        Parameters:
        u: np.ndarray
            Control input [u_x, u_y, u_theta] (accelerations in body frame)
        """
        # Split control input into linear and angular components
        linear_u = u[:2]  # [u_x, u_y]
        angular_u = u[2]  # u_theta

        # Clamp linear and angular accelerations to maximum limits
        linear_u = np.clip(linear_u, -self.max_ddx, self.max_ddx)
        angular_u = np.clip(angular_u, -self.max_ddtheta, self.max_ddtheta)

        # # Update linear velocity in the global frame
        self.v += linear_u * self.dt
        # self.v += linear_u * self.dt
        speed = np.linalg.norm(self.v)

        # Clamp linear velocity to maximum speed
        if speed > self.max_dx:
            self.v = self.v / speed * self.max_dx

        # Update position
        self.x += self.v * self.dt

        # Update angular velocity
        self.omega += angular_u * self.dt
        self.omega = np.clip(self.omega, -self.max_dtheta, self.max_dtheta)

        # Update heading angle
        self.theta += self.omega * self.dt

        # Keep heading within [-pi, pi]
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

        # Save state history
        self.x_hist = np.vstack((self.x_hist, np.array([self.x[0], self.x[1], self.theta])))