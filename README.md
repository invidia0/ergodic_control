# Ergodic Control implementation
This code implements the HEDAC and SMC ergodic control strategies.
* HEDAC 2D
* SMC 2D

Credits for the original version of the code:\
https://gitlab.idiap.ch/rli/robotics-codes-from-scratch/-/tree/master

## 1. Installation

Download the code and install the dependencies:

```bash
git clone https://github.com/invidia0/ergodic_control.git
```
```bash
cd ergodic_control
```
```bash
python3 -m venv .env
```
```bash
source .env/bin/activate
```
```bash
pip3 install -r requirements.txt
```

It's a good practice to create a virtual environment to avoid conflicts with other projects.

## 2. Quick run

* HEDAC 2D
```bash
python3 hedac.py
```

* SMC 2D
```bash
python3 smc.py
```
## Explanation (TBD)
### Parameters
| Parameter | Description               |
| --------- | ------------------------- |
| $\alpha$       | Diffusion $\rightarrow$ low values slower diffusion |
| $\beta$       | Convective heat flow $\rightarrow$ governs cooling over the whole area (increase to cool faster the area)      |
| $\gamma$      | Local cooling of agents $\rightarrow$ governs the collision avoidance (increase to push away agents) |

### HEDAC
The HEDAC implementation refers to the paper:

[1] S. Ivić, B. Crnković and I. Mezić, "Ergodicity-Based Cooperative Multiagent Area Coverage via a Potential Field," in IEEE Transactions on Cybernetics, vol. 47, no. 8, pp. 1983-1993, Aug. 2017, doi: 10.1109/TCYB.2016.2634400.

### SMC
The SMC implementation refers to the paper:

[2] George Mathew, Igor Mezić, "Metrics for ergodicity and design of ergodic dynamics for multi-agent systems," Physica D: Nonlinear Phenomena, Volume 240, Issues 4–5, 2011, Pages 432-442, ISSN 0167-2789, https://doi.org/10.1016/j.physd.2010.10.010.
