# Getting Started with qkrylov

This tutorial will guide you through the basic steps of setting up a quantum many-body simulation using `qkrylov`.

## 1. Defining the Hilbert Space

The first step in any simulation is defining the basis. `qkrylov` supports several pre-defined basis types.

```python
import qkrylov

N = 10 # Number of sites
# Create a Spin-Half basis
basis = qkrylov.SpinHalfBasis(N)

# Optionally, restrict to a specific symmetry sector (e.g., total Sz = 0)
sec = qkrylov.Sector()
sec.use_sz = True
sec.sz2 = 0 # sz2 is 2 * Sz
basis_conserved = qkrylov.SpinHalfBasis(N, sec)
```

## 2. Setting up the Sites and Operators

Each basis type is associated with a `Site` type that defines the local Hilbert space and available operators.

```python
site = qkrylov.SpinHalfSite()
# Available operators: "Sz", "Sp", "Sm", "Sx", "Sy"
```

## 3. Constructing the Hamiltonian

Hamiltonians are defined as an `OpSum` (sum of operator products).

```python
os = qkrylov.OpSum()

# Example: Nearest-neighbor Sz-Sz interaction
for i in range(N - 1):
    os += 1.0, "Sz", i, "Sz", i+1
```

## 4. Solving for the Ground State

Use the `MatrixFreeHamiltonian` to apply the operator sum to state vectors without ever storing the full matrix.

```python
H = qkrylov.MatrixFreeHamiltonian(basis, site, os)

# Find ground state energy and eigenvector using Lanczos
result = qkrylov.lanczos_ground_state(H)
print(f"E0 = {result.energy}")
```

## 5. Computing Dynamical Properties

Once you have the ground state, you can compute the dynamical structure factor $S(\omega)$.

```python
# S(omega) starting from a state phi0
phi0 = list(result.eigenvector) # Simple example
coeffs = qkrylov.continued_fraction_coeffs(H, phi0, n_iter=100)

# Evaluate at specific frequency omega
S_w = qkrylov.evaluate_spectral_function(coeffs, omega=1.5, E0=result.energy, eta=0.1)
```
