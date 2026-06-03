# Developer Guide

Welcome! To keep the project organized as it grows, please follow this directory structure for new code.

## Directory Overview

### `include/qkrylov/`
**Purpose:** C++ Header files (`.hpp`).
- All public APIs should live here.
- Organize by sub-namespace: `core/`, `basis/`, `hamiltonian/`, `linalg/`, `operators/`, `sites/`, `solvers/`, `symmetry/`.
- Use `#pragma once` for include guards.

### `src/`
**Purpose:** C++ Implementation files (`.cpp`).
- Mirror the structure of the `include/` directory.
- Keep implementations here to reduce compile times and avoid header-only bloat.

### `bindings/`
**Purpose:** Language wrappers and interface code.
- **`python/`**: Place `pybind11` or `nanobind` C++ code and Python-side wrappers here.
- **`julia/`**: Place `CxxWrap.jl` code and Julia-side package logic here.

### `tests/`
**Purpose:** Unit and integration tests.
- **C++ Tests**: Add `.cpp` files here and register them in `tests/CMakeLists.txt`.
- **Python Tests**: Place in `bindings/python/tests/` using `pytest`.
- **Julia Tests**: Place in `bindings/julia/test/` using the standard `Test` module.

### `cmake/`
**Purpose:** Build system modules.
- Put helper scripts, dependency finders, and configuration templates (`.cmake`, `.cmake.in`) here.

## General Rules

1. **Modern C++**: Use C++20 features where appropriate (`std::popcount`, concepts, etc.).
2. **Matrix-Free**: Never implement a logic that requires storing a full $N \times N$ matrix unless it's for a small-scale benchmark/test.
3. **No External Bloat**: Prefer standard library solutions. If a heavy dependency (like Eigen or Spectra) is needed, discuss it first.
4. **Build System**: All C++ files must be registered in the root `CMakeLists.txt` or a subdirectory's `CMakeLists.txt`.
