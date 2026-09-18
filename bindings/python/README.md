# `qkrylov` Python Package

The `qkrylov` Python package delivers high-performance Python bindings for the `qkrylov` C++20 matrix-free exact diagonalization and Krylov subspace library. Built on [`nanobind`](https://github.com/wjakob/nanobind) and [`scikit-build-core`](https://scikit-build-core.readthedocs.io/), it provides zero-copy NumPy array interoperability, SciPy `LinearOperator` compatibility, and seamless acceleration.

---

## Installation

### From PyPI (Standard CPU or Precompiled Wheels)
```bash
pip install qkrylov
```

### From Custom GPU Wheel Index (CUDA / ROCm)
```bash
# NVIDIA CUDA 12
pip install qkrylov --extra-index-url https://peithonking.github.io/qkrylov-wheels/cuda

# AMD ROCm 6
pip install qkrylov --extra-index-url https://peithonking.github.io/qkrylov-wheels/rocm
```

### From Source (Local Development)
Ensure you have a C++20 compiler (GCC 11+, Clang 13+) and CMake 3.20+:
```bash
pip install .
```

---

## Quickstart Example

```python
import numpy as np
import qkrylov as qk
from qkrylov.solvers import Lanczos, Davidson, FTLM

# 1. Define Hilbert space with Sz = 0 symmetry
sector = qk.Sector()
sector.set_sz(0)
basis = qk.SpinHalfBasis(6, sector)
site = qk.SpinHalfSite()

# 2. Build 1D Heisenberg model Hamiltonian terms
op = qk.OpSum()
for i in range(5):
    op += 1.0, "Sz", i, "Sz", i + 1
    op += 0.5, "Sp", i, "Sm", i + 1
    op += 0.5, "Sm", i, "Sp", i + 1

# 3. Create MatrixFreeHamiltonian
H = qk.MatrixFreeHamiltonian(basis, site, op)
print(f"Dimension: {H.dimension}")

# 4. Matrix-vector product y = H * x (zero-copy NumPy interop)
x = np.zeros(H.dimension, dtype=np.complex128)
x[0] = 1.0
y = H.apply(x)

# 5. Ground state via Lanczos (Functional & OOP)
res_gs = qk.lanczos_ground_state(H, compute_eigenvectors=True)
print(f"Ground State Energy: {res_gs.energy:.10f}")
psi0 = res_gs.eigenvector  # 1D numpy array

# 6. Excited states via Davidson
res_dav = qk.davidson_lowest(H, n_eig=3, max_subspace=20, tol=1e-8)
print(f"Lowest 3 Eigenvalues: {res_dav.eigenvalues}")

# 7. Multi-temperature thermodynamic sweep (FTLM)
betas = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
res_ftlm = qk.ftlm_sweep(H, beta_grid=betas, observables=[H], n_random=20, n_steps=40)
print(f"Specific heats: {res_ftlm.specific_heats}")
print(f"Energy expectations: {res_ftlm.observable_expectations[0]}")
```

---

## Key Features

### 1. SciPy `LinearOperator` Interoperability
`MatrixFreeHamiltonian` can be wrapped as a standard SciPy `LinearOperator` for use with `scipy.sparse.linalg` algorithms:

```python
from scipy.sparse.linalg import eigsh

A = H.as_linear_operator()
evals, evecs = eigsh(A, k=2, which='SA')
```

### 2. Dual Precision (FP64 & FP32)
`qkrylov` provides first-class support for single (`float32` / `complex64`) and double (`float64` / `complex128`) precision:
- `qkrylov.MatrixFreeHamiltonian_FP64` (default)
- `qkrylov.MatrixFreeHamiltonian_FP32`

Passing FP32 NumPy arrays automatically dispatches to the FP32 backend for GPU throughput and reduced memory footprint.

### 3. Solvers & Algorithms

| Solver | OOP Class | Functional API | Description |
| :--- | :--- | :--- | :--- |
| **Lanczos** | `qkrylov.solvers.Lanczos` | `qkrylov.lanczos_ground_state`, `qkrylov.lanczos_lowest` | Ground state ($k=1$) and lowest $k$ eigenpairs with single-pass or memory-frugal two-pass execution. |
| **Davidson** | `qkrylov.solvers.Davidson` | `qkrylov.davidson_lowest` | Subspace eigensolver with diagonal preconditioning for simultaneous multiple low-lying states. |
| **FTLM** | `qkrylov.solvers.FTLM` | `qkrylov.ftlm_sweep`, `qkrylov.ftlm` | Decoupled Finite Temperature Lanczos for multi-temperature sweeps of thermodynamic equations of state ($Z, F, E, C_v, S$) and physical observables. |
| **Continued Fraction** | `qkrylov.solvers.ContinuedFraction` | `qkrylov.continued_fraction_coeffs`, `qkrylov.evaluate_spectral_function` | Lanczos tridiagonal representation of dynamical Green's function across broad frequency windows. |
| **Correction Vector** | `qkrylov.solvers.CorrectionVector` | `qkrylov.correction_vector` | Complex shifted linear system solve $(H - E_0 - \omega - i\eta) |x\rangle = \hat{O} |\psi_0\rangle$ for targeted high-resolution spectral functions. |

---

## Running Unit Tests

Run the complete test suite using `pytest`:

```bash
pytest bindings/python/tests -v
```
