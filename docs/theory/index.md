# Theoretical Foundations

This section details the theoretical physics and numerical analysis foundations underpinning `qkrylov`.

---

- **[Exact Diagonalization](exact_diag.md)**: Exponential scaling of Hilbert spaces, quantum symmetry reduction, Hamiltonian sparsity, and the matrix-free paradigm.
- **[Krylov Subspace Methods](krylov.md)**: Mathematical derivation of the Lanczos three-term recurrence, Rayleigh-Ritz projections, Kaniel-Paige-Saad convergence bounds, and finite-precision reorthogonalization strategies.
- **[Performance & Matrix-Free Scaling](../performance.md)**: Concrete comparison of dense, CSR sparse, and matrix-free memory consumption across lattice sizes $N=12 \dots 32$, two-pass algorithms, and multi-threaded scaling.
- **[Spectral Functions & Dynamics](spectral.md)**: Zero-temperature Green's functions, continued fraction expansion, and the correction vector shifted-linear solve alternative.
- **[Finite Temperature Lanczos Method (FTLM)](../solvers/ftlm.md)**: High-temperature random trace estimation, decoupled multi-temperature sweeps, and arbitrary observable projections with error bars.
