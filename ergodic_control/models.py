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

class SecondOrderAgent:
    """
    A point mass agent with second order dynamics.
    """
    def __init__(
        self,
        x, # initial position
        max_dx=1, # maximum velocity
        max_ddx=0.2, # maximum acceleration
        dt=1, # time step
    ):
        self.x = np.array(x)  # position
        self.nbVarX = len(x)
        self.dx = np.zeros(self.nbVarX)  # velocity

        self.dt = dt  # time step

        self.max_dx = max_dx
        self.max_ddx = max_ddx

        self.x_hist = np.empty((0, self.nbVarX))

    def update(self, gradient):
        """
        set the acceleration of the agent to clamped gradient
        compute the position at t+1 based on clamped acceleration
        and velocity
        """
        ddx = gradient # we use gradient of the potential field as acceleration
        # clamp acceleration if needed
        if np.linalg.norm(ddx) > self.max_ddx:
            ddx = self.max_ddx * ddx / np.linalg.norm(ddx)

        # x = x + dt * dx + 0.5 * dt * dt * ddx (Equation of motion)
        self.x = self.x + self.dt * self.dx + 0.5 * self.dt * self.dt * ddx
        self.x_hist = np.vstack((self.x_hist, self.x))

        self.dx += self.dt * ddx  # v = v + a * dt
        # clamp velocity if needed
        if np.linalg.norm(self.dx) > self.max_dx:
            self.dx = self.max_dx * self.dx / np.linalg.norm(self.dx)

class SecondOrderAgentWithHeading:
    """
    A 2D point-mass agent with second-order dynamics, including heading.
    """
    def __init__(
        self,
        x,  # initial position [x, y]
        theta=0,  # initial heading
        max_dx=1,  # maximum velocity
        max_ddx=0.2,  # maximum acceleration
        max_dtheta=np.pi / 4,  # maximum angular velocity
        dt=1,  # time step
        id=0, # agent id
    ):
        self.x = np.array(x)  # position [x, y]
        self.theta = theta  # heading
        self.dx = 0  # velocity magnitude
        self.dtheta = 0  # angular velocity

        self.dt = dt  # time step
        self.max_dx = max_dx
        self.max_ddx = max_ddx
        self.max_dtheta = max_dtheta

        self.id = id
        self.x_hist = np.empty((0, len(x)))

    def update(self, gradient):
        """
        Update the agent's dynamics based on the gradient of the potential field.
        """
        # Compute desired heading from gradient
        theta_desired = np.arctan2(gradient[1], gradient[0])
        gradient_norm = np.linalg.norm(gradient)

        # Clamp gradient magnitude for acceleration
        acceleration = min(self.max_ddx, gradient_norm)

        # Update heading to align with gradient
        heading_error = theta_desired - self.theta
        # Ensure the error is within [-pi, pi]
        if heading_error > np.pi:
            heading_error -= 2 * np.pi
        elif heading_error < -np.pi:
            heading_error += 2 * np.pi

        self.dtheta = np.clip(heading_error / self.dt, -self.max_dtheta, self.max_dtheta)
        self.theta += self.dtheta * self.dt

        # Update velocity magnitude
        self.dx += acceleration * self.dt
        self.dx = min(self.dx, self.max_dx)

        # Compute velocity components
        # velocity = [dx * cos(theta), dx * sin(theta)]
        velocity = np.array([self.dx * np.cos(self.theta), self.dx * np.sin(self.theta)])

        # Update position
        self.x = self.x + velocity * self.dt
        self.x_hist = np.vstack((self.x_hist, self.x))

    def get_state(self):
        """
        Return the current state of the agent.
        """
        return {
            "id": self.id,
            "position": self.x,
            "heading": self.theta,
            "velocity": self.dx,
            "angular_velocity": self.dtheta,
            "trajectory": self.x_hist,
        }