# qkrylov

A modern C++20 framework for matrix-free Krylov methods in quantum many-body physics.

## Overview

`qkrylov` provides a high-performance core for performing exact diagonalization and Krylov-based calculations (like Lanczos and Davidson) without explicitly constructing Hamiltonian matrices. By implementing the matrix-free action $y = Hx$, the library enables the study of much larger Hilbert spaces than traditional matrix-based methods.

## Features Completed

- **C++20 Core**: Leveraging modern C++ features for performance and safety.
- **Basis Abstraction**: Generic basis management with support for symmetry sectors.
- **Supported Models**:
    - **Spin-Half Systems**: Heisenberg, transverse-field Ising, etc.
    - **Fermionic Systems**: Spinless fermions with Jordan-Wigner phases.
    - **Hubbard Models**: Interacting electrons with spin conservation.
    - **t-J Models**: Doped antiferromagnets with no-double-occupancy constraint.
- **Matrix-Free Hamiltonian**: Efficient application of operator sums (`OpSum`) to state vectors.
- **Advanced Solvers**:
    - **Lanczos**: Accurate ground-state energy and Ritz vector calculation.
    - **Davidson**: Iterative solver for the lowest few eigenpairs.
    - **Dynamics**: Continued Fraction Lanczos for dynamical structure factor $S(\omega)$ calculations.
- **Multi-Language Scaffolding**: Robust Python interface via `nanobind`.

## Build Requirements

- C++20 compatible compiler (e.g., GCC 11+, Clang 13+, MSVC 19.30+)
- CMake 3.20+
- `nanobind` (install via `pip install nanobind`)

## Quick Start

### Build the library and tests

```bash
make build
```

### Install Python package
```bash
pip install ./bindings/python
```

### Run tests

```bash
make test
pytest bindings/python/tests/test_basic.py
```

### C++ Example

```cpp
#include <qkrylov/basis/spinhalf_basis.hpp>
#include <qkrylov/operators/opsum.hpp>
#include <qkrylov/sites/spinhalf_site.hpp>
#include <qkrylov/hamiltonian/matrix_free_hamiltonian.hpp>
#include <qkrylov/solvers/lanczos.hpp>
#include <iostream>

using namespace qkrylov;

int main() {
    int N = 4;
    auto basis = std::make_shared<SpinHalfBasis>(N);
    auto site = std::make_shared<SpinHalfSite>();

    OpSum os;
    for (int i = 0; i < N - 1; ++i) {
        // Heisenberg interaction: Sz_i Sz_{i+1} + 0.5(Sp_i Sm_{i+1} + Sm_i Sp_{i+1})
        os += {1.0, {{"Sz", i}, {"Sz", i+1}}};
        os += {0.5, {{"Sp", i}, {"Sm", i+1}}};
        os += {0.5, {{"Sm", i}, {"Sp", i+1}}};
    }

    MatrixFreeHamiltonian H(basis, site, os);
    auto result = lanczos_ground_state(H);

    std::cout << "Ground state energy: " << result.energy << std::endl;
    return 0;
}
```

## Python Example

```python
import qkrylov

# 4-site Heisenberg chain
N = 4
basis = qkrylov.SpinHalfBasis(N)
site = qkrylov.SpinHalfSite()

os = qkrylov.OpSum()
for i in range(N - 1):
    # Heisenberg interaction: Sz_i Sz_{i+1} + 0.5(Sp_i Sm_{i+1} + Sm_i Sp_{i+1})
    os += 1.0, "Sz", i, "Sz", i+1
    os += 0.5, "Sp", i, "Sm", i+1
    os += 0.5, "Sm", i, "Sp", i+1

H = qkrylov.MatrixFreeHamiltonian(basis, site, os)
result = qkrylov.lanczos_ground_state(H)

print(f"Ground state energy: {result.energy}")
```

## Things To Be Done (Roadmap)

- **GPU Acceleration**: CUDA/HIP support for Hamiltonian application.
- **HDF5 Integration**: Efficient storage of large eigenvectors and results.
- **Julia Bindings**: Interop via `CxxWrap.jl`.
- **Finite Temperature**: Finite Temperature Lanczos Method (FTLM).

## Documentation

Full documentation is available in the `docs/` directory. See `docs/source/tutorial.md` for a comprehensive guide modeled after iTensor.
