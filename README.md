# Ergodic Control implementation
This code implements the HEDAC and SMC ergodic control strategies.
* HEDAC 2D
* SMC 2D

Credits for the original version of the code:\
https://gitlab.idiap.ch/rli/robotics-codes-from-scratch/-/tree/master

## 1. Installation

Download the code and install the dependencies:

```bash
git clone -b sitl_simulations https://github.com/invidia0/ergodic_control.git
```

Remember to create a virtual environment and install the dependencies inside it:
```bash
python3 -m venv .env
```
```bash
source .env/bin/activate
```
```bash
pip3 install -r requirements.txt
```

To run a simulation, use the following command:
```bash
python3 double_integrator.py
```

To change some parameters, you can modify the `doubleintegrator.json` file.