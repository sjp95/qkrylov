# API Reference

## Classes

### Sector
Defines symmetry sectors for basis construction.
- `use_sz`, `sz2`: Spin-Z conservation (2*Sz).
- `use_n`, `n`: Total particle number (Fermions).
- `use_nup`, `nup`, `use_ndn`, `ndn`: Spin-resolved particle numbers.

### Basis
Abstract base class for Hilbert space bases.
- `SpinHalfBasis(N, sector)`
- `FermionBasis(N, sector)`
- `HubbardBasis(N, sector)`
- `TJBasis(N, sector)`

### Site
Defines local operators for a site.
- `SpinHalfSite()`: "Sz", "Sp", "Sm", "Sx", "Sy".
- `FermionSite()`: "C", "Cdag", "N", "Id".
- `HubbardSite()`: "CUp", "CDn", "CdagUp", "CdagDn", "Nup", "Ndn", "Nupdn".
- `TJSite()`: Same as Hubbard but excludes double occupancy.

### MatrixFreeHamiltonian
Handles the action of a Hamiltonian on a vector.
- `apply(vin, vout)`: Computes $H|v_{in}\rangle$.
- `diagonal()`: Returns the diagonal of the Hamiltonian.

## Solvers

### Lanczos
- `lanczos_ground_state(H, maxiter, tol)`: Returns `LanczosResult` (energy, eigenvector).

### Davidson
- `davidson_lowest(H, n_eig, max_subspace, tol)`: Returns `DavidsonResult` (eigenvalues, eigenvectors).

### Dynamics
- `continued_fraction_coeffs(H, phi0, n_iter)`: Computes Lanczos coefficients for CF expansion.
- `evaluate_spectral_function(res, omega, E0, eta)`: Evaluates the spectral function.
