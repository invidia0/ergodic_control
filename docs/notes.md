# Project Skeye

**Description**: Create an algorithm to control a quadcopter (with a real model) that can explore an unkown environment, while avoiding obstacles. Furthermore, the algorithm should naturally balance between exploration and exploitation. Think like a quadcopter going around and while learning the map also covering the interest areas. As the quadcopter covers the area, the interest area is updated so it gets less and less interesting. If the area has a uniform distribution of interest, the quadcopter should explore the area uniformly.

**LLM for Drone-Human Interaction? 🤔**

**Example**:
In the image below, the heatmap shows areas that need patrolling.
<img width="2057" alt="image" src="../pictures/strava-heatmap-santamonica.png">

# Main approaches
### HEDAC

For an in-depth description of the HEDAC method: [HEDAC Breakdown](subpages/hedac_breakdown.md)

#### Problems
- Computationally expensive for high-dimensional spaces.
- Excessive focus on local coverage and not global coverage.
- If the gradient of the error field is zero, the agents will stop moving!
- Still no guarantee of collision avoidance.

#### Cool Stuff
- We have a **boundary condition**!. This means that the environment can have any shape (see **HEDAC Finite Element Method (FED)** for more details).

> [!NOTE]
> With the RBF we can relax the empirical distribution of the agents from being a Dirac delta function to a smooth function. This is important because with the RBF, we don't need a common basis, like the Fourier basis, to represent the coverage density. In this way we can directly compare the coverage density with the goal density.

---

### SMC
For an in-depth description of the SMC method: [SMC Breakdown](subpages/smc_breakdown.md)

### Problems
- Computationally expensive for high-dimensional spaces.
- The Fourier basis functions are not localized in space. This means that the basis functions are not zero outside of the domain $\mathcal{X}$.
- We don't have a **boundary condition**. This means that the environment has to be convex.
- No collision avoidance guarantees.

**Possible solutions**:
- This can be solved with **Laplace-Beltrami** operator considering flows on the surface. (DARS 2024)
    - **Flows** are really promising. But how to balance between global and local focus? 
    - What about the agent being the source of flows? I can assing a weight to the flows and choose the promising one based on the patrolling, like an Ergodic-MPPI.

# Notes
- Only **Stefan Ivić** has considered the problem of a different sensor model (like a FOV of a camera mounted on a drone).
- **Flows** are promising and guarantee a collision avoidance behaviour!
- Ergodic Optimization is a hot topic, because it still has a lot of issues. 
  - Soft contraints.
  - MINCO type formulation.
- Random Sampling Flows?

- Finding a way to balance local and global exploration is still a
  challenge. The paper M. Sun, A. Gaggar, P. Trautman, T. Murphey "**Fast Ergodic Search with Kernel Functions,**" tries to balance this with an hyperparameter ([Link](https://arxiv.org/abs/2403.01536)).


> [!IMPORTANT]
> **The main problem with Ergodic control and Optimization is that the cost functional is non-quadratic and does not follow the Bolza form, making optimization difficult!**  
> [1] L. M. Miller and T. D. Murphey, "**Trajectory optimization for continuous ergodic exploration,**" 2013 American Control Conference, Washington, DC, USA, 2013, pp. 4196-4201, doi: 10.1109/ACC.2013.6580484. [[**Link**](https://ieeexplore.ieee.org/document/6580484)]  
> [2] A. Mavrommati, E. Tzorakoleftherakis, I. Abraham and T. D. Murphey, "**Real-Time Area Coverage and Target Localization Using Receding-Horizon Ergodic Exploration,**" in IEEE Transactions on Robotics, vol. 34, no. 1, pp. 62-80, Feb. 2018, doi: 10.1109/TRO.2017.2766265. [[**Link**](https://ieeexplore.ieee.org/abstract/document/8114522)]  
> [3] Dong, Dayi, Henry Berger, and Ian Abraham. "**Time optimal ergodic search.**" Robotics: Science and Systems. 2023. [[**Link**](https://www.roboticsproceedings.org/rss19/p082.pdf)] 

# Breakdown of the project

- **Unknown environment**: The robot has to go around in an unexplored environment and build its own map. Need to manage **frontier-based** exploration inside Ergodic exploration.
> [!TIP]  
> **Flow-based** exploration is promising. Probably next step.

- **Unknown information pdf** 
    - Time-Varying Gaussian Processes? **Uncertainty and Mean estimates**.
        - The uncertainty can be used to explore the environment. 
        - The mean can be used to exploit the environment.

        I can achieve a "common" distribution density by combining the mean and uncertainty.
    - The problem with learning the environemnt with a Gaussian Process is that I also have to consider my trajectory empirical distribution to be time-varying. This is because the agent is moving and the environment is changing.

- **Balance between global and local coverage**
    - **RBF (HEDAC) + Gaussian Processes**. 
    - **Kernel Functions Ergodic Control**
        - **Problem**: Optimization to find the hyperparams (like in a Gaussian Process).
- Include a FOV-like sensor as "cooling" or RBF like in the HEDAC. 

# RSS

Try to address the problem of HEDAC by using a combination of Mean and Uncertainty of the Gaussian Process. The HEDAC has the problem to focus too much on local coverage and not global coverage. So we can drive the agent to explore the environment by using the uncertainty of the Gaussian Process. While the mean can be used to focus on the areas of interest and exploit the RBF local behavior. In this way we don't need to have an already known pdf and, moreover, we can ensure a global behaviour with a time-varying uncertainty from the Gaussian Processes. 

To address the **HEDAC problem**, which tends to focus excessively on local coverage at the expense of global coverage, we propose a solution that leverages both the **mean** and **uncertainty** from a Gaussian Process (GP):  

- The **uncertainty** is used to guide the agent's exploration of the environment, ensuring global coverage.  
- The **mean** directs the agent toward areas of interest, effectively exploiting the **local behavior** of Radial Basis Functions (RBF).  
- This approach eliminates the need for a predefined probability density function (PDF), while the time-varying uncertainty from the GP ensures adaptability to changing conditions, promoting both exploration and exploitation.
- In this way, the agents has always a gradient guiding him around the environment. This is also a problem of the HEDAC because if the gradient is zero, the agents will stop moving.

### Summary

- **Problem**:  
  - HEDAC focuses too much on **local coverage** and neglects **global coverage**.  

- **Proposed Solution**:  
  - Combine the **mean** and **uncertainty** of the Gaussian Process.  
    - **Uncertainty**: Drives global exploration of the environment.  
    - **Mean**: Targets areas of interest for local exploitation, leveraging RBF properties.  
  - No need for a predefined PDF.  
  - Adaptable to dynamic environments via the time-varying uncertainty of the GP.



<!-- ### Modeling the Environment  
- Use **Gaussian Processes (GPs)** to model the environment's underlying distribution.  
  - Combine mean and uncertainty to balance **exploration** and **exploitation**.

### Exploring and Exploiting the Environment  
- Utilize **Spectral Multiscale Coverage (SMC)** for global coverage behavior.  
- Consider **Radial Basis Functions (RBF)** for their smoothing behavior.  

### Trajectory Planning  
- Adopt a **receding horizon approach** for trajectory planning:  
  - Plan for a fixed time horizon and then replan.  
- Design a trajectory cost inspired by **EGO-Planner**:  
  - Include **Time-Optimality** and **Ergodicity** as soft constraints.  

### Time-Varying Distribution  
- Address the **time-varying aspect** of the distribution in future modeling.   -->


## Roadmap for RSS
- [x] Check what happens with only the RBF and without the heat equation (in different environments).
  - **Done.** The RBF has the sigma parameter to tune the smoothness of the function. This is a drawback because the sigma parameter is not easy to tune and introduces a new parameter.
  - **Problem:** Boundary conditions are not considered, how to avoid the agents to go outside the environment?
  - Source term is not **smoothed**, should I smooth it with a RBF again?
  - Instead of istantly zeroing the informative pdf, try to make it decay over time as the agents spend time in the area?
- [x] Add the RBF for the environment.
- [x] Add the Gaussian Process for the environment and find a way to combine the mean and uncertainty.
  - **Done.** Procedure to combine the mean and uncertainty:
    - Mean and Uncertainty are normalized (min-max).
    - Negative values are removed.
    - Mean and Uncertainty are combined with a sum.
    - The combined value is **normalized over the domain**.
- [x] Add a model with the heading and a **FOV-like sensor**.
- [x] Add GPs inside the HEDAC algorithm. 
  - [x] What if instead of fitting with all the samples, I fit only the samples that matters? -> ICRA 2025 Paper
  - [x] MaxPooling? -> Yes this is a good idea. Works.
  - [ ] Instead of learning the full environment, I can learn a latent space and then use the latent space to reconstruct the environment. (I learn the fourier basis and then use the fourier basis to reconstruct the environment).
- [x] **Add a non-convex environment**.
  - Done. The FEM is needed to solve the heat equation in a non-convex environment where there are circular obstacles. Skip for the moment.
- [x] Add Mantovani2024 RAL filter.
- [x] Add GPs in a non-convex environment.
- [x] Add a **time-varying** aspect to the empirical distribution of the agents. This is because otherwise the “learnt” information in the environment (mean and uncertainty) won't heat inside the heat equation. \
<img width="500" alt="image" src="../pictures/source_term_problem.png">

  resolved by adding a time-varying aspect to the empirical distribution of the agents. This is the result: \
<img width="500" alt="image" src="../pictures/coverage_decay.png">

- [x] Correct the Ergodic Metric to account for the time-varying aspect of the empirical distribution of the agents.
  - **Done.** The Ergodic Metric is now correct and accounts for the time-varying aspect of the empirical distribution of the agents. \
<img width="800" alt="image" src="../pictures/ergodic_metric.png"> \
  We have normalized the coverage density and the goal density to be between 0 and 1. Then we subtract the coverage density from the goal density. This is the error field. Then we take only the positive values of the error field and this becomes the source term of the heat equation.

- [ ] Add a growing factor on the source strength or on the diffusion coefficient? Because I want the agents to learn the environment and as they get stuck in an area, I want the other areas that need attention to "grow" in importance -> Hotter areas. How to balance this?
  - Not implemented yet and not sure aboiut the implementation.
- [x] Add multiple agents.
- [ ] Add a Time-Varying Gaussian Process (?). Future Work!
- [x] Adjust the kernel clamping for a non convex map (with Shapely like with the FOV)
- [x] Local cooling? Implemented but I don't know if it works, the paper is sus.
- [x] Add samples sharing between agents and remove common GP?