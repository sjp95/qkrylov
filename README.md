# qkrylov

A modern C++20 framework for matrix-free Krylov methods in quantum many-body physics.

## Overview

`qkrylov` provides a high-performance core for performing exact diagonalization and Krylov-based calculations (like Lanczos and Davidson) without explicitly constructing Hamiltonian matrices. By implementing the matrix-free action $y = Hx$, the library enables the study of much larger Hilbert spaces than traditional matrix-based methods.

## Features Completed

- **C++20 Core**: High-performance implementation leveraging modern C++ features.
- **Matrix-Free Hamiltonian**: Efficient application of operator sums (`OpSum`) to state vectors.
- **Supported Models**:
    - **Spin-Half**: Heisenberg, transverse-field Ising, etc.
    - **Fermion**: Spinless fermions with Jordan-Wigner phases.
    - **Hubbard**: Interacting electrons with spin conservation.
    - **t-J Model**: Doped antiferromagnets with no-double-occupancy constraint.
- **Symmetry Conservation**: Support for $S^z$, total particle number $N$, and spin-resolved numbers $N_{\uparrow}/N_{\downarrow}$ sectors.
- **Advanced Solvers**:
    - **Lanczos**: Accurate ground-state energy and Ritz vector (eigenvector) calculation.
    - **Davidson**: Iterative solver for the lowest few eigenpairs of large matrices.
    - **Dynamics**: Continued Fraction Lanczos for dynamical structure factor $S(\omega)$ calculations.
- **Python Bindings**: Full library interface available in Python via `nanobind`.

## Installation

### Prerequisites
- C++20 compatible compiler
- CMake 3.20+
- Python 3.10+ (for Python bindings)
- `nanobind` (install via `pip install nanobind`)

### Build the library and tests
```bash
make build
```

### Install Python package
```bash
pip install -e ./bindings/python
```

## Testing

The library includes a comprehensive suite of tests to ensure mathematical correctness.

### C++ Tests
Run all 14 C++ unit tests:
```bash
make test
```

### Python Tests
Run Python integration tests using `pytest`:
```bash
pytest bindings/python/tests/test_basic.py
```
There are 4 specialized Python tests verifying:
1. Basic basis construction (`SpinHalf`).
2. Fermion basis with particle number conservation.
3. Hamiltonian action and diagonal calculation.
4. t-J basis construction and constraints.

## Python Example

```python
import qkrylov

# 4-site Heisenberg chain
basis = qkrylov.SpinHalfBasis(4)
site = qkrylov.SpinHalfSite()
os = qkrylov.OpSum()

for i in range(3):
    # Sz_i Sz_{i+1} + 0.5(Sp_i Sm_{i+1} + Sm_i Sp_{i+1})
    os.add_term(...) # See examples/heisenberg_python.py

H = qkrylov.MatrixFreeHamiltonian(basis, site, os)
result = qkrylov.lanczos_ground_state(H)
print(f"Ground state energy: {result.energy}")
```

## Things To Be Done (Roadmap)

- **GPU Acceleration**: CUDA/HIP support for Hamiltonian application.
- **HDF5 Integration**: Efficient storage of large eigenvectors and results.
- **Extended Symmetries**: Support for point groups and non-abelian symmetries.
- **Finite Temperature**: Implementation of the Finite Temperature Lanczos Method (FTLM).

## Documentation

Full documentation is available in the `docs/` directory. See `docs/source/tutorial.md` for a comprehensive guide.
