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
        self.x = np.array(x)  # position
        self.theta = theta  # heading
        self.dx = 0  # velocity magnitude
        self.dtheta = 0  # angular velocity

        self.dt = dt  # time step
        self.max_dx = max_dx
        self.max_ddx = max_ddx
        self.max_dtheta = max_dtheta

        self.id = id
        self.x_hist = np.empty((0, 3)) # X, Y, Theta

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
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi


        self.dtheta = np.clip(heading_error / self.dt, -self.max_dtheta, self.max_dtheta)
        self.theta += self.dtheta * self.dt
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi  # Keep theta within [-pi, pi]

        # Update velocity magnitude
        self.dx += acceleration * self.dt
        self.dx = min(self.dx, self.max_dx)

        # Compute velocity components
        # velocity = [dx * cos(theta), dx * sin(theta)]
        velocity = np.array([self.dx * np.cos(self.theta), self.dx * np.sin(self.theta)])

        # Update position
        self.x = self.x + velocity * self.dt
        self.x_hist = np.vstack((self.x_hist, np.array([self.x[0], self.x[1], self.theta])))

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

class DoubleIntegratorAgent:
    def __init__(
        self,
        x,  # initial position [x, y]
        theta=0,  # initial heading
        max_dx=1,  # maximum velocity
        max_ddx=0.2,  # maximum acceleration
        max_dtheta=np.pi / 4,  # maximum angular velocity
        max_ddtheta=np.pi / 8,  # maximum angular acceleration
        dt=1,  # time step
        id=0,  # agent id
    ):
        self.x = np.array(x, dtype=np.float64)  # Position [x, y]
        self.v = np.array([0.0, 0.0], dtype=np.float64)  # Velocity [vx, vy]
        self.theta = theta  # Heading angle (yaw)
        self.omega = 0.0  # Angular velocity

        self.max_dx = max_dx  # Maximum linear velocity
        self.max_ddx = max_ddx  # Maximum linear acceleration
        self.max_dtheta = max_dtheta  # Maximum angular velocity
        self.max_ddtheta = max_ddtheta  # Maximum angular acceleration

        self.dt = dt  # Time step
        self.id = id  # Agent ID

        self.x_hist = np.empty((0, 3))  # History of states [x, y, theta]

    def update(self, u):
        """
        Update the agent state using control input:
        u[0:2] - linear acceleration in world frame [ax, ay]
        u[2]   - angular acceleration (yaw)
        """
        a = np.array(u[:2])
        alpha = float(u[2])

        # Clamp linear acceleration
        norm_a = np.linalg.norm(a)
        if norm_a > self.max_ddx:
            a = self.max_ddx * a / norm_a

        # Clamp angular acceleration
        alpha = np.clip(alpha, -self.max_ddtheta, self.max_ddtheta)

        # Update velocity and clamp
        self.v += a * self.dt
        speed = np.linalg.norm(self.v)
        if speed > self.max_dx:
            self.v = self.max_dx * self.v / speed

        # Update position
        self.x += self.v * self.dt

        # Update angular velocity and clamp
        self.omega += alpha * self.dt
        self.omega = np.clip(self.omega, -self.max_dtheta, self.max_dtheta)

        # Update heading
        self.theta += self.omega * self.dt
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta)) # Normalize angle

        # Log state
        self.x_hist = np.vstack((self.x_hist, [self.x[0], self.x[1], self.theta]))

    def track_velocity_and_heading(
        self,
        v_target_world,
        theta_target=None,
        kp_lin=1.0,
        kp_lat=0.1,
        kp_theta=2.0,
        kd_theta=1.0,
        penalize_lateral=True,
    ):
        """
        Track desired velocity (in world frame) and optionally heading.

        v_target_world: array_like, shape (2,) - desired velocity [vx, vy] in world frame
        theta_target: float or None - desired heading (yaw), None to skip heading control
        kp_lin: float - gain on forward velocity error
        kp_lat: float - gain on sideways velocity suppression
        penalize_lateral: bool - if True, suppress sideways motion
        """

        vx_d, vy_d = v_target_world

        # Rotation matrix (world to body)
        c, s = np.cos(self.theta), np.sin(self.theta)
        R = np.array([[c, s], [-s, c]])

        v_body = R @ self.v
        v_d_body = R @ np.array([vx_d, vy_d])

        # Control in body frame
        a_body = np.zeros(2)
        a_body[0] = kp_lin * (v_d_body[0] - v_body[0])  # longitudinal
        if penalize_lateral:
            a_body[1] = -kp_lat * v_body[1]  # drive lateral to 0

        # Convert back to world frame
        a_world = R.T @ a_body

        # Heading control (PD)
        if theta_target is not None:
            e_theta = np.arctan2(np.sin(theta_target - self.theta),
                                 np.cos(theta_target - self.theta))
            alpha = kp_theta * e_theta - kd_theta * self.omega
        else:
            alpha = 0.0

        # Apply control
        u = np.hstack([a_world, alpha])
        self.update(u)

    def get_acceleration(
        self,
        v_target_world,
        theta_target=None,
        kp_lin=1.0,
        kp_lat=0.1,
        kp_theta=2.0,
        kd_theta=1.0,
        penalize_lateral=True,
    ):
        """
        Track desired velocity (in world frame) and optionally heading.

        v_target_world: array_like, shape (2,) - desired velocity [vx, vy] in world frame
        theta_target: float or None - desired heading (yaw), None to skip heading control
        kp_lin: float - gain on forward velocity error
        kp_lat: float - gain on sideways velocity suppression
        penalize_lateral: bool - if True, suppress sideways motion
        """

        vx_d, vy_d = v_target_world

        # Rotation matrix (world to body)
        c, s = np.cos(self.theta), np.sin(self.theta)
        R = np.array([[c, s], [-s, c]])

        v_body = R @ self.v
        v_d_body = R @ np.array([vx_d, vy_d])

        # Control in body frame
        a_body = np.zeros(2)
        a_body[0] = kp_lin * (v_d_body[0] - v_body[0])  # longitudinal
        if penalize_lateral:
            a_body[1] = -kp_lat * v_body[1]  # drive lateral to 0

        # Convert back to world frame
        a_world = R.T @ a_body

        # Heading control (PD)
        if theta_target is not None:
            e_theta = np.arctan2(np.sin(theta_target - self.theta),
                                 np.cos(theta_target - self.theta))
            alpha = kp_theta * e_theta - kd_theta * self.omega
        else:
            alpha = 0.0

        # return control
        u = np.hstack([a_world, alpha])
        return u