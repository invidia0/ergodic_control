# Metrics for ergodicity and design of ergodic dynamics for multi-agent systems

**Paper**:

*Mathew, G. and Mezić, I.*, 2011. **Metrics for ergodicity and design of ergodic dynamics for multi-agent systems**. Physica D: Nonlinear Phenomena, 240(4-5), pp.432-442. [[**Link**](https://www.sciencedirect.com/science/article/pii/S016727891000285X)]

## Spectral Multiscale Coverage (SMC)

We have a domain:

$$\mathcal{X}=[0,L_1]\times\cdots\times[0,L_d], $$  

where $L_i$ is the boundary length of the $i$-th dimension.
- $x = [x_1, \dots, x_d] \in \mathcal{X}$ is a state in the search space.
- $k = [k_1, \dots, k_d]$ are a Fourier basis function, where each entry of the vector is a whole number $k_i \in [0, 1, 2, \dots, K]$

**Normalized Fourier Basis Function**:
$$
\begin{align}
f_k(x) & = \frac{1}{h_k} \prod_{i=1}^{d} \cos\left( \frac{k_i \pi}{L_i} x_i \right),
\end{align}
$$

where $h_k$ is the normalization constant:
$$
\begin{align}
h_k = \left( \int_0^{L_1} \int_0^{L_2} \cos^2(k_1 x_1) \cos^2(k_2 x_2) \, dx_1 \, dx_2 \right)^{1/2}
\end{align}
$$

**Change of basis**

Given a spatial probability density function $p(x)$, we can transform this function as follow:

$$
\begin{align}
    p(x) & = \sum_{k}^{\infty} \left( \underbrace{\int_{\mathcal{X}} p(x) f_k(x) dx}_{\phi_k} \right) \cdot f_k(x) \\
        & = \sum_{k}^{\infty} \phi_k \cdot f_k(x)
\end{align}
$$

where all the coefficients $[\phi_k]$ are the coordinate of the function $p(x)$ under the bases $[f_k(x)]$.

### Ergodic metric
Represents the distance measure between a trajectory and a distribution.

This is the **empirical distribution** of the trajectory of an agent:
$$
\begin{align}
\Phi_{s(t)}(x) = \frac{1}{T} \int_{0}^{T} \delta(x {-} s(t)) dt,
\end{align}
$$

Now let's change basis:
$$
\begin{align}
    \Phi_{s(t)}(x) & = \sum_{k}^{\infty} \left[ \int_{\mathcal{X}} \left( \frac{1}{T} \int_{0}^{T} \delta(x {-} s(t)) dt \right) f_k(x) dx \right] \cdot f_k(x) \\
    & = \sum_{k}^{\infty} \left(\underbrace{\frac{1}{T} \int_{0}^{T} f_k(s(t)) dt}_{c_k} \right) \cdot f_k(x) \\
        & = \sum_{k}^{\infty} c_k \cdot f_k(x)
\end{align}
$$

We now have the spatial distribution $p(x)$ and the empirical distribution $\Phi_{s(t)}(x)$ in the same finite dimensional vector space. 

The ergodic metric is defined as the distance between these two distributions:
$$
\begin{align}
    \mathcal{E}(p(x), s(t)) & = \sum_{k} \lambda_k \cdot (c_k - \phi_k)^2, \\
    \lambda_k & = (1 + \Vert k \Vert)^{-\frac{d+1}{2}}.
\end{align}
$$

### First Order Ergodic Dynamics
- First order dynamics: $\dot{x}(t) = u_j(t).$
- Second order dynamics: $\ddot{x}(t) = u_j(t).$

#### At a given time t, let us solve the optimal control problem over the time horizon $[t, t + \Delta t], \tau=t+\Delta t$.
The cost-function we are going to use is the first time-derivative of $$\Phi(\tau)$$ at the end of the horizon. i.e., we
aim to drive the agents to positions which lead to the highest rate
of decay of the coverage metric.
$$
\Phi(t) := \frac{1}{2} N^2 t^2 \mathcal{E}(t).
$$

**Cost function**:
$$
C(t, \Delta t) = \dot{\Phi}(t + \Delta t)
$$

$$
\begin{align}

u^*_j(\tau) &= \arg \min_{\|u_j(\tau)\|^2 \leq u_{\max}} H(x, S, W, u, \tau)\\
&= -u_{\max} \frac{\beta_j(\tau)}{\|\beta_j(\tau)\|^2}, \text{ if } \beta_j(\tau) \neq 0,
\end{align}
$$
and where $\beta_j(\tau) = \gamma_j(\tau) + \sum_{k} \sigma_k(\tau) \nabla f_k(x_j(\tau))$.

For $\Delta t \rightarrow 0 $ then, at time $t$ with the system state $x(t)$, we can compute the control signal $u(t)$ as:
$$
\begin{align}
    u(t) & = -u_d \cdot \frac{b(t)}{\vert b(t) \vert} \\
    b(t) & = \sum_{k} \lambda_k \cdot \left( \frac{1}{t} \int_{0}^{t} f_k(x(\tau)) d\tau -\phi_k \right) \cdot \nabla f_k(x(t)) \\
    \nabla f_k(x(t)) & = \begin{bmatrix} -\frac{k_1\pi}{L_1 h_k} \cdot \sin\left( \frac{k_1 \pi}{L_1} x_1 \right) \cos\left( \frac{k_2 \pi}{L_2} x_2 \right) \cdots \cos\left( \frac{k_d \pi}{L_d} x_d \right) \\ \vdots \\ -\frac{k_i\pi}{L_i h_k} \cdot \cos\left( \frac{k_1 \pi}{L_1} x_1 \right) \cdots \sin\left( \frac{k_i \pi}{L_i} x_i \right) \cdots \cos\left( \frac{k_d \pi}{L_d} x_d \right) \\ \vdots \\ -\frac{k_d\pi}{L_d h_k} \cdot \cos\left( \frac{k_1 \pi}{L_1} x_1 \right) \cdots \cos\left( \frac{k_{d-1} \pi}{L_{d-1}} x_{d-1} \right) \sin\left( \frac{k_d \pi}{L_d} x_d \right) \end{bmatrix} \in \mathbb{R}^d , 
\end{align}
$$

where $u_d$ is the maximum control input. With this, we have a closed-form solution for the optimal ergodic control input, that ensures the system to have ergodic dynamics.

## Problems
- Computationally expensive for high-dimensional spaces.
- The Fourier basis functions are not localized in space. This means that the basis functions are not zero outside of the domain $\mathcal{X}$.
- We don't have a **boundary condition**. This means that the environment has to be convex.
- No collision avoidance guarantees.