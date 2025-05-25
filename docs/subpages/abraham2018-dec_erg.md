# Decentralized Ergodic Control: Distribution-Driven Sensing and Exploration for Multiagent Systems

**Paper**:

*I. Abraham and T. D. Murphey,* "**Decentralized Ergodic Control: Distribution-Driven Sensing and Exploration for Multiagent Systems**," in IEEE Robotics and Automation Letters, vol. 3, no. 4, pp. 2987-2994, Oct. 2018, doi: 10.1109/LRA.2018.2849588. [[**Link**](https://ieeexplore.ieee.org/document/8392363)]

## Decentralized Ergodic Control
Consider a nonlinear dynamical system with a state $x(t) \in \mathbb{R}^n$ and control input $u(t) \in \mathbb{R}^m$:
$$
\begin{equation}
\dot{x}(t) = f(x(t), u(t)) = g(x(t)) + h(x(t)) u(t)
\end{equation}
$$

where $g(x) : \mathbb{R}^n \to \mathbb{R}^n$ and $h(x) : \mathbb{R}^n \to \mathbb{R}^{n \times m}$ are known functions.
They consider the same Ergodic metric as in the SMC method (see [SMC](../subpages/smc_breakdown.md)):
$$
\begin{align}
    \mathcal{E}(x(t)) & = q\sum_{k \in N^v} \Lambda_k \cdot (c_k - \phi_k)^2, \\
    \Lambda_k & = (1 + \Vert k \Vert)^{-\frac{d+1}{2}}.
\end{align}
$$

### Reciding Horizon Ergodic Control
A history of where a robot has been is maintained in memory in
order to compute the ergodic metric, since the target distribution $\phi(s)$ can be time-varying. They add a term $\Delta t$ to the ergodic metric to control how far in the past the robot should remember where it has been.
$$
\begin{equation}
\mathcal{E}(x(t)) = q \sum_{k \in N^v} \left( \frac{1}{T_\mathcal{E}} \int_{t_i - \Delta t_\mathcal{E}}^{t_i + T} F_k (x(t)) \, dt - \phi_k \right)^2.
\end{equation}
$$

#### Ergodic controller

We would like to minimize $(4)$ with respect to $x(t)$ and $u(t)$, like in the iLQR method, i.e., we have an agent with an horizon and we would like to find in this horizon the best trajectory to minimize the ergodic metric.

**This is really hard to solve in real-time! Let's see why in the table below.**
#### Problems

| **Problem** | **Description** |
|-------------|-----------------|
| **Non-Quadratic Cost Functional** | The ergodic cost is non-quadratic and does not follow the Bolza form, making optimization difficult. |
| **High Computational Cost of Trajectory Optimization** | Infinite-dimensional trajectory optimization methods are computationally prohibitive for real-time control. |
| **Complexity of Quadratic Reformulation** | Change of coordinates can make the cost functional quadratic, but it increases the state space, affecting real-time performance. |
| **Inefficiency in Integration of Dynamic Trajectories** | Running costs on information states result in repetitive integration, reducing efficiency. |
| **Stability Issues with Terminal Cost-Only Optimization** | Optimizing only terminal cost complicates ensuring the stability of MPC algorithms. |

> [!IMPORTANT]
> **The main problem with Ergodic control and optimization is that the cost functional is non-quadratic and does not follow the Bolza form, making optimization difficult!**  
> [1] L. M. Miller and T. D. Murphey, "**Trajectory optimization for continuous ergodic exploration,**" 2013 American Control Conference, Washington, DC, USA, 2013, pp. 4196-4201, doi: 10.1109/ACC.2013.6580484. [[**Link**](https://ieeexplore.ieee.org/document/6580484)]  
> [2] A. Mavrommati, E. Tzorakoleftherakis, I. Abraham and T. D. Murphey, "**Real-Time Area Coverage and Target Localization Using Receding-Horizon Ergodic Exploration,**" in IEEE Transactions on Robotics, vol. 34, no. 1, pp. 62-80, Feb. 2018, doi: 10.1109/TRO.2017.2766265. [[**Link**](https://ieeexplore.ieee.org/abstract/document/8114522)]  
> [3] Dong, Dayi, Henry Berger, and Ian Abraham. "**Time optimal ergodic search.**" Robotics: Science and Systems. 2023. [[**Link**](https://www.roboticsproceedings.org/rss19/p082.pdf)] 

<!-- They consider the **sensitivity** of the ergodic metric with respect to an infinitesimal time of application $\lambda \in \mathbb{R}^+ \rightarrow 0$ of the **best possible input** $u^*(t)$ that sufficiently reduces the ergodic metric $\mathcal{E}(x(t))$ at time $\tau$ from some default control input $u_{def}(t)$. -->
- **Objective**: Analyze the **sensitivity** of the ergodic metric ($ \mathcal{E}(x(t)) $) with respect to:  
  - An infinitesimal time of application ($ \lambda \to 0 $, where $ \lambda \in \mathbb{R}^+ $).  

- **Inputs Considered**:  
  - **Default control input**: $ u_{def}(t) $.  
  - **Optimal control input**: $ u^*(t) $, which sufficiently reduces $ \mathcal{E}(x(t)) $ at time $ \tau $.  

- **Key Idea**: Evaluate the impact of applying $ u^*(t) $ for a very short duration ($ \lambda $) on the reduction of $ \mathcal{E}(x(t)) $.  

**First-order sensitivity**: they take the derivative of (4) with respect to the duration
time $\lambda$ of control $u_*(t)$.
$$
\begin{equation}
\frac{\partial \mathcal{E}}{\partial \lambda} \bigg|_{\tau} = \rho(\tau)^\intercal \left( f_2 (\tau, \tau) - f_1 (\tau) \right),
\end{equation}
$$
where:
$$
\begin{align}
    f_2 (t, \tau) & = f(x(t), u(\tau)), \\
    f_1 (t) & = f(x(t), u_{def}(t)), \\
    \dot{\rho} &= -2 q \sum_{k \in N_v} \Lambda_k (c_k - \phi_k) \frac{\partial F_k}{\partial x} - \frac{\partial f}{\partial x} \rho(t)
\end{align}
$$

Given the mode insertion gradient, we seek to find the control
$u_*(t)$ that most significantly decreases in the objective $(4)$.
$$
\begin{equation}
J_2 = \int_{t_i}^{t_i + T} \frac{\partial \mathcal{E}}{\partial \lambda} \bigg|_t + \frac{1}{2} \| u_*(t) - u_{\text{def}}(t) \|^2_R,
\end{equation}
$$
the matrix $R$ weights the control effort. The solution to this optimization problem is the control $u_*(t)$ given by:
$$
\begin{equation}
u_*(t) = -R^{-1} h(x) \rho(t) + u_{\text{def}}(t).
\end{equation}
$$

Ok so now we have the best control input $u_*(t)$ that reduces the ergodic metric $\mathcal{E}(x(t))$. We need to find the **best time to apply this control input**. In this work, they select a time of application $\tau$ that results in the most negative mode insertion gradient:
$$
\begin{equation}
\tau_* = \arg \min_{\tau} \frac{\partial \mathcal{E}}{\partial \lambda}.
\end{equation}
$$

Finally, we need to find the **best duration of application** $\lambda$ of the control input $u_*(t)$ that reduces the ergodic metric $\mathcal{E}(x(t))$ the most. The resulting control is then added to the default control $u_{\text{def}}(t) = u_*(\tau) \forall t \in [\tau, \tau + \lambda] \cap [t_i, t_i + t_s]$ where $t_s$  is the sampling time and $u_*(\tau)$ is saturated.

## Decentralized Ergodic Control

**Consensus on the fourier coefficients**

**TLDR**: They use a consensus algorithm to make the robots agree on the Fourier coefficients of the target distribution.


### Problems
- They use GPs but they do not exploit the uncertainty of the environment. They only consider the mean of the GP. ([Video](https://www.youtube.com/watch?v=Jibt4GLj5sw))
<img width="2057" alt="image" src="../../pictures/never-explored-area.png">