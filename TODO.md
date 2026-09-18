# qkrylov Roadmap & TODO List

## 1. Python Binding Safety Hardening
- [x] Add array dimension validation in `bindings/python/src/binding.cpp` for `MatrixFreeHamiltonian::apply` (throw clean Python exception if array size != `H.dimension()` instead of segfaulting).
- [x] Add tuple length validation in `OpSum.__iadd__` in `binding.cpp` to guard against malformed tuple indexing.
- [x] Wrap raw pointer allocation in `vec_to_numpy` with `std::unique_ptr` prior to `nb::capsule` creation to prevent memory leaks on exception.

## 2. Documentation System (Zensical)
- [x] Set up `zensical.toml` configuration and install `zensical` in `.venv`.
- [x] Configure `.github/workflows/docs.yml` for automated deployment to GitHub Pages using Zensical.
- [x] Populate `docs/` with quickstart guide, API references, solver guides (Lanczos, Davidson, Dynamics, FTLM), and zero-copy performance tips.

## 3. Modernized Solvers & Decoupled FTLM Architecture
- [x] Unified C++20 Lanczos compile-time policies (`OnePass`, `OnePass_DKGS`, `OnePass_full`, `TwoPass`) with structured bindings and `LanczosConfig`.
- [x] Decoupled FTLM into sample collection (`ftlm_sample`) and multi-temperature sweep evaluation (`ftlm_evaluate_sweep`) with arbitrary physical observable projection $\mathcal{O}_{jk} = \langle v_j | \hat{O} | v_k \rangle$ and full thermodynamic equations of state ($Z, F, E, C_v, S$).
- [x] Flat C ABI endpoints for two-pass Lanczos, lowest-$k$ Lanczos, and FTLM sweeps (`qkrylov_ftlm_sweep_fp64`/`_fp32`).
- [x] Julia SciML `CommonSolve` interface (`solve(prob, alg)`) with `GroundStateProblem`, `ExcitedStatesProblem`, `ThermalProblem`, `DynamicalProblem`, `CorrectionVectorProblem`.
- [x] Nanobind Python bindings with SciPy `LinearOperator` interoperability, dual-precision dispatch (FP64/FP32), OOP solvers (`Lanczos`, `Davidson`, `FTLM`, `ContinuedFraction`, `CorrectionVector`), and comprehensive test suite (134 tests).
- [x] Numerical stability fixes: energy variance clamp $\ge 0$ for high-$\beta$ low-temperature FTLM sweeps in single-precision (FP32).

## 4. Complete GitHub Actions & `.github` Setup
- [x] Finalize `.github/workflows/pypi_publish.yaml` for PyPI wheel release builds (Linux x86_64, Linux ARM64, Windows, Mac Intel, Mac Apple Silicon).
- [x] Configure PyPI index publishing & GitHub Pages redirect index generation (`--extra-index-url` PyTorch-style setup).
- [x] Add `.github/workflows/tests.yaml` for fast CI test suite execution on push/PR.
- [x] Add Issue Templates and Pull Request Templates under `.github/`.

## 5. Universal `extern "C"` API Layer for Julia and even more languages
- [x] Create `include/qkrylov/c_api.h` exposing flat `extern "C"` functions for Basis, Site, OpSum, MatrixFreeHamiltonian, Solvers, BLAS-1 kernels, and Device Vectors.
- [x] Implement `src/c_api.cpp` to bridge C calls to internal C++ classes without copying array buffers.
- [x] Support seamless zero-copy interoperability for Julia (`ccall`), Rust (`bindgen`), C, and Go.

## 6. Multi-Hardware Distribution & Shipping Strategy (Target Architecture)
*(Note for C++ engineers: This is the final shipping structure. Ensure the Kokkos integration and CMake configuration natively support this CI/CD matrix).*

### A. The Pre-Compiled Binaries Matrix
We will ship a total of 18 pre-compiled wheels across Python 3.11 and 3.12+.

**1. OS & Architecture Matrix (The 18 Wheels)**

| Architecture / Backend | Linux (python 3.11) | Linux (python 3.12+) | Win (python 3.11) | Win (python 3.12+) | Mac (python 3.11) | Mac (python 3.12+) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CPU (x86)** | PyPI | PyPI | PyPI | PyPI | * | * |
| **CPU (ARM)** | PyPI | PyPI | * | * | PyPI | PyPI |
| **GPU (CUDA 12)** | GitHub | GitHub | GitHub | GitHub | | |
| **GPU (CUDA 11)** | * | * | * | * | | |
| **GPU (ROCm 6)** | GitHub | GitHub | * | * | | |
| **GPU (ROCm 5)** | * | * | * | * | | |
| **GPU (SYCL / Intel)** | GitHub | GitHub | GitHub | GitHub | | |

*\* Hardware combination exists, but we do not ship pre-compiled wheels. Compile via Docker Builder (or from source).*

**2. Distribution Payload Specs**

| Target Hardware | Backend | Host Location | Wheel Size |
| :--- | :--- | :--- | :--- |
| **Basic CPUs** | CPU (OpenMP) | **PyPI** | ~1-3 MB |
| **NVIDIA GPUs** | CUDA 12 | **GitHub Releases** | ~1-3 MB* |
| **AMD GPUs** | ROCm 6 | **GitHub Releases** | 50-200 MB** |
| **Intel GPUs** | SYCL | **GitHub Releases** | 50-200 MB** |

*\*CUDA wheels are kept tiny because `auditwheel` is configured to exclude NVIDIA libraries, relying instead on `nvidia-*` PyPI dependencies.*
*\*\*AMD and Intel wheels are built as "Fat Wheels" (bundling the massive runtimes inside the wheel via `auditwheel`) to ensure a seamless plug-and-play experience without forcing users to install system-level toolkits.*

### B. User Installation Experience
Users install based on their hardware with a single command. No compilation required.

- **Standard CPU / Mac Users:** (Installs from PyPI)
  `pip install qkrylov`
- **NVIDIA Users (CUDA):** (Installs from GitHub custom index)
  `pip install qkrylov --extra-index-url https://peithonking.github.io/qkrylov-wheels/cuda`
- **AMD Users (ROCm):**
  `pip install qkrylov --extra-index-url https://peithonking.github.io/qkrylov-wheels/rocm`
- **Intel iGPU/dGPU Users (SYCL):**
  `pip install qkrylov --extra-index-url https://peithonking.github.io/qkrylov-wheels/sycl`

### C. Edge Devices (The Docker "Builder Pattern")
For users on non-standard hardware (e.g., Jetson Nano, Raspberry Pi 5) or legacy enterprise clusters (e.g., CUDA 11), we **do not** cross-compile binaries. Instead, we distribute a Docker container designed purely as an isolated compiler factory.

**The Edge-User Workflow:**
1. The user runs the builder image to compile a perfectly tuned binary for their local machine, mapping their host directory as the output target:
   `docker run --rm -v $(pwd):/output peithonking/qkrylov-builder:latest build --backend=cuda`
2. The Docker container outputs a compiled `qkrylov.whl` into their host directory and cleanly destroys itself.
3. The user installs the native wheel into their standard host Python environment:
   `pip install ./qkrylov.whl`

*(End goal: Maximum hardware utilization with zero host-OS toolchain pollution for the user).*
