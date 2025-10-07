[![python](https://img.shields.io/badge/python-3.12+-informational.svg)](https://www.python.org/downloads/)

# Learning to Satisfy Constraints while Boosting the Performance with ADMM

Python implementation of ADMM for Performance Boosting
with Constraint Satisfaction as described in the article
"Learning to Satisfy Constraints while Boosting the Performance with ADMM" and [2] with Control Barrier Functions (CBFs).

## Project Setup

In order to run this project it is recommended to use a virtual environment
and required to install some dependencies. 
The setup process for both actions is automated using `setup.py`.

### Installation

Follow these steps to set up the project:

#### 1. Clone the repository
```bash
git clone https://github.com/DecodEPFL/NNs-for-OC
cd Safe_Performance_Boosting
```

#### 2. Run the setup script
Execute the following command to create a virtual environment and install dependencies:
```bash
python setup.py
```

#### 3. Activate the virtual environment
After installation, activate the environment (if working on the console):
- **On macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```
- **On Windows (CMD or PowerShell):**
  ```powershell
  venv\Scripts\activate
  ```
Otherwise, activate the environment through the UI 
of your preferred development environment (PyCharm, VS Code, etc).

### Dependencies
The project requires the following dependencies, which are automatically installed from `requirements.txt`:
- `torch`
- `numpy`
- `matplotlib`
- `jax`
- `pip`
- `tqdm`
- `cvxpy`
- `clarabel`

## Benchmark case study: a point-mass robot

This system is described by the following difference equation

```math
\begin{align}
    x_{t} \!&=\! x_{t-1}\! +\! T_s\! \!\begin{bmatrix}
        q_{t-1}\\
        \!M^{-1}(\beta_1q_{t-1}\!\!+\!\beta_2 \mathrm{tanh}(\!q_{t-1}\!)\!+\!F_{t-1}\!)\!
    \end{bmatrix}\! +w_t,\!\!\\
    F_t\!&=F'_t(a_t)+u_t,
\end{align}
```
with $x_t=\begin{bmatrix}a_t^\top & q_t^\top\end{bmatrix}^\top$ its state composed by the Euclidian positions $a_t$ and velocities $q_t$ , respectively, and the input voltage and $u_t$ the performance boosting input. 

The following gif showcases the performance boosting with velocity constraints achieved with the safe performance boosting algorithm.

<p align="center">
     <img src="gifs/IO_tr_C12.gif" alt="iLasso-DeePC trajectory tracking, data selection, and BPIs">
</p> 

## License
This project is licensed under the terms of the `CC-BY-4.0` license.
See [LICENSE](LICENSE) for more details.


## References
- 
[[2]](https://arxiv.org/pdf/2405.00871) Luca Furieri, Clara Galimberti, Giancarlo Ferrari Trecate.
"Learning to boost the performance of stable nonlinear closed-loop systems," 2024.

