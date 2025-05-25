# Second Order Agent Model with Heading

The second order agent model with heading is a model that extends the second order agent model by adding a heading to the agent. The heading is the direction in which the agent is facing. The agent can move in the direction of its heading and change its heading based on its interactions with the environment.

## State Variables
We have the following state variables for the second order agent model with heading:
$$
\begin{equation}
\mathbf{x} = \begin{bmatrix} x \\ y \\ v \\ \theta \end{bmatrix}
\end{equation}
$$

where:
- $x$ and $y$ are the agent's position in the environment. Let's consider $\mathbf{z} = [x, y]$.
- $v$ is the agent's velocity.
- $\theta$ is the agent's heading.

## Potential Field
We want to move our agent with respect to a potential field, described by:
$$
\begin{equation}
\nabla u(\mathbf{z}, t)
\end{equation}
$$

## Dynamics
The dynamics of the second order agent model with heading are given by the following equations:
$$
\begin{equation}
\begin{aligned}
\dot{x} &= v \cos(\theta) \\
\dot{y} &= v \sin(\theta) \\
\dot{v} &= a \\
\dot{\theta} &= \omega
\end{aligned}
\end{equation}
$$

where:
- $a$ is the acceleration of the agent.
- $\omega$ is the angular velocity of the agent.

### Position Dynamics
The position dynamics of the second order agent model with heading are given by:
$$
\begin{equation}
\begin{aligned}
\dot{x} &= v \cos(\theta) \\
\dot{y} &= v \sin(\theta)
\end{aligned}
\end{equation}
$$

### Velocity Dynamics
The velocity dynamics of the second order agent model with heading are given by:
$$
\begin{equation}
\dot{v} = a
\end{equation}
$$

where:
- $a$ is the acceleration of the agent.

### Heading Dynamics
The heading dynamics of the second order agent model with heading are given by:
$$
\begin{equation}
\theta_{des} = \arctan2(\nabla u_y, \nabla u_x)
\end{equation}
$$

where:
- $\theta_{des}$ is the desired heading of the agent.
- $\nabla u_x$ and $\nabla u_y$ are the components of the gradient of the potential field.

The angular velocity of the agent is given by:
$$
\begin{equation}
\omega = k (\theta_{des} - \theta)
\end{equation}
$$

where:
- $k$ is a proportional gain.

## Control Inputs
The control inputs for the second order agent model with heading are:
$$
\begin{equation}
\mathbf{u} = \begin{bmatrix} a \\ \omega \end{bmatrix}
\end{equation}
$$

## Control Policy
### 1. Position update
The position of the agent is updated based on its velocity and heading:
$$
\begin{equation}
\begin{aligned}
x(t+\Delta t) &= x_t + v_t \cos(\theta_t) \Delta t \\
y(t+\Delta t) &= y_t + v_t \sin(\theta_t) \Delta t
\end{aligned}
\end{equation}
$$

### 2. Velocity update
The velocity of the agent is updated based on its acceleration:
$$
\begin{equation}
v(t+\Delta t) = v_t + a_t \Delta t,
\end{equation}
$$
the velocity magnitude is limited by a maximum value: $v(t) = \min(\|v(t)\|, v_{\text{max}})$.
Where $a_t$ is the acceleration at time $t$, derived from the potential field $\nabla u(\mathbf{z}, t)$:
$$
\begin{equation}
a_t = \min(\|\nabla u(\mathbf{z}, t)\|, a_{\text{max}}),
\end{equation}
$$

and $a_{\text{max}}$ is the maximum acceleration.

### 3. Heading update
The desired heading of the agent is calculated based on the gradient of the potential field:
$$
\begin{equation}
\theta_{\text{des}} = \arctan2(\nabla u_y, \nabla u_x),
\end{equation}
$$

where $\nabla u_x$ and $\nabla u_y$ are the components of the gradient of the potential field.

The heading error is calculated as:
$$
\begin{equation}
\theta_{\text{error}} = \theta_{\text{des}} - \theta,
\end{equation}
$$

and the angular velocity of the agent is updated based on the heading error:
$$
\begin{equation}
\dot{\theta} = \frac{\theta_{\text{error}}}{\Delta t},
\end{equation}
$$

where $\Delta t$ is the time step.
The angular velocity is clipped by a maximum value: $\omega(t) = \text{clip}(\omega(t), -\omega_{\text{max}}, \omega_{\text{max}})$.
This because we want to avoid the agent to turn too fast.
Finally, the heading of the agent is updated based on its angular velocity:
$$
\begin{equation}
\theta(t+\Delta t) = \theta_t + \omega_t \Delta t.
\end{equation}
$$






