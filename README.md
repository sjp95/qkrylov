# qkrylov

A modern C++20 framework for matrix-free Krylov methods in quantum many-body physics.

## Overview

`qkrylov` provides a high-performance core for performing exact diagonalization and Krylov-based calculations (like Lanczos) without explicitly constructing Hamiltonian matrices. By implementing the matrix-free action $y = Hx$, the library enables the study of much larger Hilbert spaces than traditional matrix-based methods.

## Current Features

- **C++20 Core**: Leveraging modern C++ features for performance and safety.
- **Basis Abstraction**: Generic basis management with support for symmetry sectors.
- **Spin-Half Systems**: Complete implementation of `SpinHalfBasis` and local operators (`Sz`, `Sp`, `Sm`, `Sx`, `Sy`).
- **Matrix-Free Hamiltonian**: Efficient application of operator sums (`OpSum`) to vectors.
- **Lanczos Solver**: Basic ground-state Lanczos implementation.
- **Multi-Language Scaffolding**: Prepared structures for Python and Julia bindings.

## Build Requirements

- C++20 compatible compiler (e.g., GCC 11+, Clang 13+, MSVC 19.30+)
- CMake 3.20+
- (Optional) `make` for convenience

## Quick Start

### Build the library and tests

```bash
make build
```

### Run tests

```bash
make test
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
        os.add_term({1.0, {{"Sz", i}, {"Sz", i+1}}});
        os.add_term({0.5, {{"Sp", i}, {"Sm", i+1}}});
        os.add_term({0.5, {{"Sm", i}, {"Sp", i+1}}});
    }

    MatrixFreeHamiltonian H(basis, site, os);
    auto result = lanczos_ground_state(H);

    std::cout << "Ground state energy: " << result.energy << std::endl;
    return 0;
}
```

## Project Status

This project is in **pre-alpha**. The core C++ logic is functional for spin-half systems. We are currently working on:
- Davidson and Jacobi-Davidson solvers.
- Fermionic and Bosonic site types.
- Python bindings via `nanobind` or `pybind11`.
- Julia bindings via `CxxWrap.jl`.
- GPU acceleration (CUDA/HIP).

For the long-term vision and planned API examples, see [VISION.md](./VISION.md).
