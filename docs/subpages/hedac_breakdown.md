# Ergodicity-Based Cooperative Multiagent Area Coverage via a Potential Field

**Paper**:

*S. Ivić, B. Crnković and I. Mezić*, "**Ergodicity-Based Cooperative Multiagent Area Coverage via a Potential Field**," in IEEE Transactions on Cybernetics, vol. 47, no. 8, pp. 1983-1993, Aug. 2017, doi: 10.1109/TCYB.2016.2634400. [[**Link**](https://ieeexplore.ieee.org/document/7786872)]

## Heat Equation Driven Area Coverage (HEDAC)

**SMC problems that they try to address**:

Use of the Fourier transform in the SMC algorithm can be computationally demanding and in some special cases cause
instability in agent movement. Due to the choice of the **global**
Fourier basis, the method tends to perform global coverage
first and later focuses on local coverage. In some applications, it is more advantageous to explore the local area first.
Furthermore, in SMC, excessively covered areas can cause
suboptimal behavior of agents.

**HEDAC Approach**:

We present a new method which is able to carry out a cooperative and synchronized multiagent movement with a task to achieve targeted spatial area coverage, focusing on local exploration. This is achieved by first formulating the ergodicity problem in terms of **—local—radial basis functions (RBFs)** and formulating a **“smoothed gradient potential field”** control method to minimize the error between the delta measure on trajectories and the prescribed measure. Both smoothing of the gradient and collision avoidance (that is not built into the basic SMC method) are achieved using a heat equation formulation.

- RBFs
- Smoothed gradient potential field with heat equation formulation.
---
### Description

- *n*-dimensional domain $\Omega \subset \mathbb{R}^n$.
- $\mathbf{z}_i:[0, t] \rightarrow \mathbb{R}^n$ for $i = 1, \dots, N$ are the **trajectories** of the $N$ agents.

Smooth Local RBF:
$$
\begin{align}
\phi_\sigma (x) &= \sigma^{-n} \phi \left( \frac{x}{\sigma} \right)\\
\int_{\mathbb{R}^n} \phi(x) \, dx &= 1
\end{align}
$$
and $\sigma$ is a scaling factor. What the RBF does is to smooth the gradient of the coverage density.

**Coverage density at time $t$**:
$$
\begin{equation}
\tilde{c}_{\sigma} (\mathbf{x}, t) = \frac{1}{N t} \sum_{i=1}^{N} \int_{0}^{t} \phi_{\sigma} (\mathbf{x} - \mathbf{z}_i(\tau)) \, d\tau. \tag{1}
\end{equation}
$$
Normalized coverage density:
$$
\begin{equation}
c_{\sigma} (\mathbf{x}, t) = \frac{\tilde{c}_{\sigma} (\mathbf{x}, t)}{\int_{\Omega} \tilde{c}_{\sigma} (\mathbf{x}, t) \, d\mathbf{x}}.
\end{equation}
$$

Let's define the **goal density** as $m:\Omega \rightarrow \mathbb{R}$, such that $\int_{\Omega} m(\mathbf{x}) \, d\mathbf{x} = 1$. Now we can smooth the goal density and obtain the **spatial diffusion** of $m$:

$$
\begin{equation}
(\phi_\sigma \ast m)(x) = \int_{\mathbb{R}^n} \phi_\sigma(y) m(x - y) \, dy = \int_{\mathbb{R}^n} \phi_\sigma(x - z) m(z) \, dz
\end{equation}
$$

This operation essentially “spreads” the goal density $m$ over the domain $\Omega$. This smoothing process helps to distribute the goal density more evenly across the domain, ensuring that the agents can cover the area more effectively.

> [!NOTE]
> In the code, the smoothing of the goal density is done by employing Fourier basis functions. Recostruncting the goal density from a limited number of Fourier coefficients is equivalent as applying a low-pass filter to the goal density, resulting in a smoothed version of the goal density based on the number of Fourier coefficients used.

#### Connection with SMC
The SMC coverage is defined as:
$$
\begin{equation}
c_{\text{smc}}(x) = \frac{1}{N t} \sum_{i=1}^{N} \int_{0}^{t} \delta(x - z_i(\tau)) \, d\tau
\end{equation}
$$
But this is really similar to $(3)$, where the delta function is replaced by the RBF. In order to have SMC and RBF to be comparable, we need to have:
$$
\begin{equation}
\lim_{\sigma \to 0} \phi_\sigma (x) = \lim_{\sigma \to 0} \sigma^{-n} \phi \left( \frac{x}{\sigma} \right) = \delta(x)
\end{equation}
$$

Particularly, we have:

$$
\begin{align}
\lim_{\sigma \to 0} c_\sigma (x) &= c_{\text{smc}}(x), \\
\lim_{\sigma \to 0} (\phi_\sigma \ast m)(x) &= m(x).
\end{align}
$$

It can be seen that for a small $\sigma$ the coverage $c_{\sigma}$ and $\phi_\sigma \ast m$ are reasonable approximations of $c_{smc}$ and $m$, respectively. For a small $\sigma$ it is possible to compare the convergence results for SMC and HEDAC algorithms.

> [!NOTE]
> With the RBF we can relax the empirical distribution of the agents from being a Dirac delta function to a smooth function. This is important because with the RBF, we don't need a common basis, like the Fourier basis, to represent the coverage density. In this way we can directly compare the coverage density with the goal density.

### Ergodic metric
The local error is the difference between goal and coverage density:
$$
\begin{equation}
e(x, t) = (\phi_\sigma \ast m)(x) - c_\sigma (x, t)
\end{equation}
$$
The scalar field $e$ is a **spatial distribution of error**, where negative and positive values indicate insufficiently and excessively
covered areas, respectively. Taking the $L_2$ norm of $e$ gives the **ergodic metric**:
$$
\begin{equation}
E(t) = \| e(x, t) \|_2
\end{equation}
$$

And we would like the system to be ergodic, i.e., $\lim_{t \to \infty} E(t) = 0$.

## Agent Motion Control and Heat Equation

**Kinematic model**:
$$
\begin{equation}
\frac{d \mathbf{z}_i(t)}{d t} = v_a \cdot \frac{\nabla u(\mathbf{z}_i(t), t)}{\|\nabla u(\mathbf{z}_i(t), t)\|}, \quad i = 1, \ldots, N
\end{equation}
$$
where $v_a$ is the agent velocity and $u$ is the potential field. The direction of the agent depends on current position of the agent and the gradient of scalar field $u$.

**Why not $u=e$?**:
- **Different Scales of Spatial Error**: Variations in error scales make direct minimization difficult.
- **Presence of Local Minima**: Prevents global minimization of the overall error $E(t)$.
- **Gradient Limitations**: The gradient of $e$ does not provide information about distant insufficiently covered areas.
- **Zero-Gradient Issue**: Encountering a zero gradient of $e$ can stop local minimization progress.
- **Need for Global Control**: Additional global strategies are required to overcome these issues, as highlighted in related work.

#### Stationary Heat Equation as Smoothing Operator for the Spatial Error Field $e$

The heat conduction phenomena within the heat
equation propagates information on the insufficiently covered
area, as temperature, throughout the whole domain.

**Heat Equation**:
$$
\begin{equation}
\frac{\partial u}{\partial t} = \alpha \Delta u(\mathbf{x}, t) - \beta u(\mathbf{x}, t) - \gamma a(\mathbf{x}, t) + s(\mathbf{x}, t),
\end{equation}
$$

that can be solved numerically by the **finite element method** [[**Link**](https://levelup.gitconnected.com/solving-2d-heat-equation-numerically-using-python-3334004aa01a)].

**Stationary Heat Equation**:
$$
\begin{align}
\frac{\partial u}{\partial t} &= 0, \\
\alpha \Delta u(\mathbf{x}) &= \beta u(\mathbf{x}) + \gamma a(\mathbf{x}) - s(\mathbf{x}),
\end{align}
$$

**Boundary Condition**:
$$
\begin{equation}
\frac{\partial u}{\partial \mathbf{n}} = 0, \quad \text{on} \, \partial \Omega
\end{equation}
$$

The **Neumann boundary condition**, represents an ideal heat insulation on the edge of the domain. Insulation is an appropriate boundary condition because it prevents agents from leaving the domain, disables the heat flux trough the boundary and it ensures that agent motion is dominated by the properties of the heat source and not by the boundary conditions.

| Symbol | Description |
| --- | --- |
| $\alpha$ | Diffusion coefficient (smoothing). Represents the thermal diffusivity which regulates the strength of the smoothing of the potential field by conduction. |
| $\beta$ | Absorption coefficient (decay of the potential field) |
| $\gamma$ | Cooling coefficient (collision avoidance between agents) |
| $\Delta$ | Laplacian operator |

**Source term**:
$$
\begin{align}
\tilde{s}(\mathbf{x}, t) &= \max(e(\mathbf{x}, t), 0)^{2}, \\
s(x, t) &= \frac{\tilde{s}(x, t)}{\frac{1}{|\Omega|}\int_{\Omega}\tilde{s}(x, t) \, dx}.

\end{align}
$$

The source term is used to propagate the information on insufficiently covered areas throughout the domain.

**Heat conduction**:
$$
\begin{equation}
\alpha \Delta u(\mathbf{x}, t)
\end{equation}
$$
A stronger heat conduction, caused by choosing larger $\alpha$, extends the range of influence of the heat (error) source in the domain and it is suitable to enhance global coverage control.

**Convective heat flow**:
$$
\begin{equation}
\beta u(\mathbf{x}, t)
\end{equation}
$$

When increasing convective cooling, temperature field u tends to have more similar shape to source field $s$ so details of uncovered areas are better emphasized in $u$. Thus, enlarging $\beta$ leads to better local coverage.

**Cooling term**:
$$
\begin{align}
a(x, t) &= \frac{\tilde{a}(x, t)}{\frac{1}{|\Omega|}\int_{\Omega}\tilde{a}(x, t) \, dx}, \\
\tilde{a}(x, t) &= \sum_{i=1}^{N} \phi_{\sigma_a} (\mathbf{x} - \mathbf{z}_i(t)).
\end{align}
$$

This is another RBF that is used to provide local repulsion effect between agents. The RBF can be the same as the coverage RBF, but it can also be different.

**DOESN'T GUARANTEE COLLISION AVOIDANCE**.


## Problems
- Computationally expensive for high-dimensional spaces.
- Excessive focus on local coverage and not global coverage.
- If the gradient of the error field is zero, the agents will stop moving!
- Still no guarantee of collision avoidance.
- **We need to have the pdf already known.**

## Cool Stuff
- We have a **boundary condition**!. This means that the environment can have any shape (see **HEDAC Finite Element Method (FED)** for more details).