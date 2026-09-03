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
    - **Lanczos**: Accurate ground-state energy, iterations/convergence tracking, and Ritz vector calculation.
    - **Davidson**: Iterative solver for the lowest $k$ eigenpairs with convergence diagnostics.
    - **Dynamics**: Continued Fraction Lanczos for dynamical structure factor $S(\omega)$ calculations.
    - **Finite Temperature**: Finite Temperature Lanczos Method (FTLM) for thermodynamic quantities ($Z, E, C_v$).
- **Multi-Language Support**: Robust Python interface via `nanobind` and native Julia package [`QuantumKrylov.jl`](bindings/julia/README.md) backed by C ABI (`c_api.h`) and prebuilt `qkrylov_jll` binary artifacts.

## Build Requirements

- C++20 compatible compiler (e.g., GCC 11+, Clang 13+, MSVC 19.30+)
- CMake 3.20+
- `nanobind` (install via `pip install nanobind`)

## Quick Start

### For Julia Users

#### Option 1: Install the latest prebuilt release (`julia-latest`)
Automatically downloads and configures the latest native prebuilt binaries (`libqkrylov.so`, `libqkrylov.dylib`, or `qkrylov.dll`). On Linux systems with an NVIDIA GPU and driver 12+, it automatically downloads the **CUDA 12 accelerated** binary with zero manual compilation:
```julia
using Pkg
Pkg.add(url="https://github.com/sjp95/qkrylov.git", rev="julia-latest", subdir="bindings/julia")
```

#### Option 2: Pin to a specific historical build
Every CI build is permanently archived with its own tagged release. To install an exact, pinned build, specify its commit SHA (e.g. `rev="1d00f75"`):
```julia
using Pkg
Pkg.add(url="https://github.com/sjp95/qkrylov.git", rev="<commit-sha>", subdir="bindings/julia")
```

#### Option 3: Julia General Registry (once registered)
```julia
using Pkg
Pkg.add("QuantumKrylov")
```

### For Python Users
```bash
pip install qkrylov
```

### For C++ Developers & Local Building
If you are developing or modifying the C++ core engine:

```bash
make build
make test
pytest bindings/python/tests/test_basic.py
julia --project=bindings/julia -e 'using Pkg; Pkg.test()'
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

### Python Example

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

### Julia Example

```julia
using QuantumKrylov

# 4-site 1D Heisenberg chain
N = 4
basis = SpinHalfBasis(N)

op = OpSum()
for i in 0:(N - 2)
    # Heisenberg interaction: Sz_i Sz_{i+1} + 0.5(Sp_i Sm_{i+1} + Sm_i Sp_{i+1})
    # Note: `global` is needed when running as a top-level script, but can be
    # omitted if this loop is inside a function or run directly in the REPL.
    global op += 1.0 * Sz(i) * Sz(i + 1) + 0.5 * (Sp(i) * Sm(i + 1) + Sm(i) * Sp(i + 1))
end

# Construct MatrixFreeHamiltonian (site model automatically inferred from basis)
# Targets GPU if available ("cuda:0"), otherwise falls back to CPU
target_device = is_gpu_build() ? "cuda:0" : "cpu"
H = MatrixFreeHamiltonian(basis, op; device=target_device)

# Compute ground state energy and wavefunction
res = lanczos_ground_state(H, return_state=true)

println("Execution device:    ", target_device)
println("Ground state energy: ", res.energy)
println("Iterations executed: ", res.iterations)
println("Convergence status:  ", res.converged)
```

## Things To Be Done (Roadmap)

- **Distributed Multi-GPU**: Multi-node MPI + CUDA/HIP Kokkos execution space scaling for very large Hilbert spaces.
- **HDF5 Integration**: Efficient storage of large eigenvectors and Krylov subspace results.

## Documentation

- **C API Reference (`extern "C"`)**: [`docs/api/c_api.md`](docs/api/c_api.md)
- **Julia Core Concepts Guide**: [`bindings/julia/documentation.md`](bindings/julia/documentation.md)
- **Julia Package README**: [`bindings/julia/README.md`](bindings/julia/README.md)

