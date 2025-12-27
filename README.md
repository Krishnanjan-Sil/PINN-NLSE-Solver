# PINN-NLSE-Solver: Solving the Non-Linear Schrödinger Equation using JAX

This repository contains a high-performance implementation of a **Physics-Informed Neural Network (PINN)** designed to solve the **Non-Linear Schrödinger (NLS)** equation.

By leveraging **JAX** for hardware acceleration and automatic differentiation, this model learns to approximate the complex-valued wave function $h(t, x)$ by minimizing a loss function based on governing physical laws, rather than relying solely on labeled data.

## 🌌 Overview

The Non-Linear Schrödinger equation is a fundamental equation in physics, essential for modeling wave propagation in:
* **Nonlinear Optics**: Pulse propagation in optical fibers.
* **Bose-Einstein Condensates**: Macroscopic quantum phenomena.
* **Plasma Physics**: Langmuir waves and soliton stability.

### The Physics
We solve for the complex field $h(t, x) = u(t, x) + i v(t, x)$ governed by:

$$i h_t + 0.5 h_{xx} + |h|^2 h = 0$$

The PINN is trained to satisfy this PDE residual across a spatio-temporal domain $(t, x) \in [0, \pi/2] \times [-5, 5]$, starting from a **Bright Soliton** initial condition.

## 🛠 Features

* **JAX & Optax**: Built with Google's JAX library for XLA-accelerated performance on GPUs and TPUs.
* **Automatic Differentiation**: Uses `jax.grad` to compute exact partial derivatives for the PDE residual, eliminating discretization errors common in Finite Difference Methods.
* **Mesh-free Training**: Learns the solution on a continuous domain using randomly sampled collocation points.
* **Complex Field Modeling**: Utilizes a dual-output MLP architecture to represent the real ($u$) and imaginary ($v$) components of the wave.

## 🚀 Getting Started

### Prerequisites
The code is designed to be lightweight and portable. It requires:
* Python 3.7+
* `jax`
* `optax`
* `matplotlib`
* `numpy`

### Installation & Usage
The easiest way to run this code is via **Google Colab**:

1.  **Clone or Download**: Get the `.ipynb` file from this repository.
2.  **Upload to Colab**: Open [Google Colab](https://colab.research.google.com/) and upload the notebook.
3.  **Enable GPU**: Go to `Runtime` > `Change runtime type` > select **T4 GPU**.
4.  **Run All**: Execute the cells. The script handles all dependencies automatically.

Alternatively, for local execution:
```bash
pip install jax jaxlib optax matplotlib numpy
python pinn_nls.py

```

## 📊 Results

The model outputs the magnitude of the wave function . After training, you can expect:

* **High Fidelity**: Close agreement with the analytical Bright Soliton solution.
* **Generalization**: The network accurately predicts the phase and amplitude even at time steps it wasn't explicitly supervised on.

## ⚖️ Copyright and License

**Copyright © 2025 Krishnanjan Sil**

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

### License Summary

* **Commercial Use**: Allowed.
* **Modification**: Allowed.
* **Distribution**: Allowed.
* **Network Service Disclosure**: If you run a modified version of this software as a network service (e.g., a web app), you **must** make your modified source code available to users.
* **Same License**: Derivative works must be licensed under AGPL-3.0.

For the full text, please see the LICENSE file.

---

*If you find this repository useful for your research, please consider starring it!*

## 📜 Citation

If you use this code in your research or project, please cite it as follows:

```bibtex
@software{Krishnanjan_Sil_PINN_NLSE_2025,
  author = {Sil, Krishnanjan},
  title = {PINN-NLSE-Solver: Physics-Informed Neural Networks for the Non-Linear Schrödinger Equation},
  url = {[https://github.com/](https://github.com/)[Krishnanjan_Sil]/[https://github.com/Krishnanjan-Sil/PINN-NLSE-Solver]},
  version = {1.0.0},
  year = {2025}
}
```
### **Contact**

For suggestions or collaboration, feel free to reach out to me:
Krishnanjan Sil – 1krishnanjansil1@gmail.com
