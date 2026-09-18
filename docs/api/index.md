# API Reference

This section provides auto-generated API documentation for the qkrylov Python package.

!!! note "Under Construction"
    Full API reference pages are being generated. Check back soon.

## Overview

The Python bindings expose all core qkrylov functionality with zero-copy memory semantics — NumPy arrays passed to C++ functions are **never copied**.

### Basis Classes

| Class | Description |
|-------|-------------|
| `SpinHalfBasis` | Hilbert space for spin-1/2 systems |
| `FermionBasis` | Hilbert space for spinless fermions |
| `HubbardBasis` | Hilbert space for Hubbard model (up + down electrons) |
| `TJBasis` | Hilbert space for t-J model |

### Hamiltonian

| Class | Description |
|-------|-------------|
| `MatrixFreeHamiltonian` | Matrix-free operator: no matrix ever stored in RAM |

### Solvers

| Function | Description |
|----------|-------------|
| `lanczos_ground_state` | Ground state via Lanczos iteration |
| `davidson_lowest` | Low-lying eigenvalues via Davidson algorithm |
| `continued_fraction_coeffs` | Lanczos coefficients for spectral functions |
| `evaluate_spectral_function` | Evaluate A(ω) from Lanczos coefficients |
| `ftlm` | Finite Temperature Lanczos Method |

---

## Language Interfaces & C API Reference

- **Julia API**: See [Julia API Reference](julia_api.md) or [`bindings/julia/README.md`](https://github.com/sjp95/qkrylov/blob/main/bindings/julia/README.md).
- **C API (`extern "C"`)**: See [C API Reference](c_api.md).



