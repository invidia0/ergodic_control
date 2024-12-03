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
    ):
        self.x = np.array(x)  # position
        # determine which dimesnion we are in from given position
        self.nbVarX = len(x)
        self.dx = np.zeros(self.nbVarX)  # velocity

        self.t = 0  # time
        self.dt = 1  # time step

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
        
        self.t += 1
        self.dx += self.dt * ddx  # v = v + a * dt
        # clamp velocity if needed
        if np.linalg.norm(self.dx) > self.max_dx:
            self.dx = self.max_dx * self.dx / np.linalg.norm(self.dx)