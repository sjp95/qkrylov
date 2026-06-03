# qkrylov

A modern matrix-free exact diagonalization and Krylov-method framework for quantum many-body physics.

## Motivation

Most exact diagonalization packages are either:

* difficult to extend,
* CPU-only,
* model-specific,
* or not designed for modern GPU architectures.

The goal of **qkrylov** is to provide an ITensor-like user experience while performing matrix-free exact diagonalization, Lanczos, Davidson, Jacobi-Davidson, finite-temperature Lanczos, and dynamical correlation calculations.

The framework is designed for:

* Spin systems
* Fermionic systems
* Hubbard models
* t-J models
* Bosonic systems
* Arbitrary many-body operator strings

without constructing the Hamiltonian matrix explicitly.

Instead of storing

H(i,j)

the framework only implements

y = Hx

which enables much larger Hilbert spaces.

---

# Design Goals

## User Friendly

The user should describe physics, not implementation details.

Example:

```python
from qkrylov import *

N = 24

sites = SpinHalf(
    N=N,
    conserve_sz=True,
    sz_sector=0
)

os = OpSum()

for i in range(N-1):
    os += 1.0, "Sz", i, "Sz", i+1
    os += 0.5, "Sp", i, "Sm", i+1
    os += 0.5, "Sm", i, "Sp", i+1

H = MatrixFreeHamiltonian(
    os,
    sites
)

result = lanczos(
    H,
    nev=4
)

print(result.eigenvalues)
```

---

# Supported Site Types

## Spin Systems

```python
SpinHalf(...)
SpinOne(...)
SpinThreeHalf(...)
Spin(S)
```

Supported operators:

```text
Sz
Sp
Sm
Sx
Sy
```

---

## Fermionic Systems

```python
Fermion(...)
Hubbard(...)
```

Operators:

```text
CdagUp
CUp
CdagDn
CDn
Nup
Ndn
Nupdn
```

---

## t-J Systems

```python
tJ(...)
```

Local states:

```text
|0>
|↑>
|↓>
```

---

## Bosonic Systems

```python
Boson(...)
```

Operators:

```text
Bdag
B
N
```

---

# Quantum Number Conservation

A major design goal is symmetry reduction.

Supported conservation laws:

## Spin Systems

```python
conserve_sz=True
```

Example:

```python
SpinHalf(
    N=24,
    conserve_sz=True,
    sz_sector=0
)
```

---

## Hubbard Systems

```python
conserve_nup=True
conserve_ndn=True
```

Example:

```python
Hubbard(
    N=12,
    nup=6,
    ndn=6
)
```

---

## Bosons

```python
conserve_particles=True
```

---

## No Symmetry

Example:

```python
Kitaev(...)
```

or

```python
SpinHalf(
    N=24,
    conserve_sz=False
)
```

---

# Matrix-Free Hamiltonian

The Hamiltonian is never stored explicitly.

Instead:

```cpp
H.apply(x,y)
```

computes

```text
y = Hx
```

directly.

Advantages:

* lower memory
* larger Hilbert spaces
* GPU friendly
* ideal for Krylov methods

---

# General Operator Strings

The framework supports arbitrary operator products.

Examples:

Two-body:

```python
os += J, "Sz", i, "Sz", j
```

Three-body:

```python
os += K,
      "Sx", i,
      "Sy", j,
      "Sz", k
```

Four-body:

```python
os += Jring,
      "Sp", i,
      "Sm", j,
      "Sp", k,
      "Sm", l
```

This design naturally supports:

* Dzyaloshinskii-Moriya interactions
* Scalar chirality
* Ring exchange
* Correlated hopping
* Custom user-defined operators

---

# Solvers

## Lanczos

Ground state and low-energy spectrum.

```python
result = lanczos(H)
```

---

## Davidson

Many low-lying eigenstates.

```python
result = davidson(H)
```

---

## Jacobi-Davidson

Large numbers of eigenpairs.

```python
result = jacobi_davidson(H)
```

---

# Dynamical Correlations

Zero-temperature spectra:

```python
S = dynamical_structure_factor(
        H,
        psi0,
        operator="Sz",
        q=np.pi
)
```

Implemented using continued-fraction Lanczos.

---

# Finite Temperature

Finite Temperature Lanczos Method (FTLM)

```python
thermal = ftlm(
    H,
    beta=10.0,
    nrandom=64
)
```

Supported observables:

* Partition function
* Energy
* Specific heat
* Susceptibility
* Dynamical correlation functions

---

# Storage

All results are stored in HDF5.

```python
save("results.h5", result)
```

Stored information:

```text
basis
sectors
eigenvalues
eigenvectors
spectra
metadata
```

---

# Parallelization

## CPU

OpenMP

## GPU

CUDA

Future:

* HIP
* SYCL

---

# Language Support

Core:

```text
C++20
```

Interfaces:

```text
Python
Julia
C++
```

Python bindings:

```text
pybind11
```

Julia bindings:

```text
CxxWrap
```

---

# Current Status

Implemented:

* Basis abstraction
* SpinHalf basis
* Symmetry sectors
* Operator strings
* OpSum
* Local spin operators
* Matrix-free Hamiltonian action

In Progress:

* Lanczos solver

Planned:

* Davidson
* Jacobi-Davidson
* FTLM
* GPU backend
* HDF5
* Python interface
* Julia interface

---

# Project Philosophy

Physics first.

The user specifies:

* Hilbert space
* Operators
* Symmetries

The framework handles:

* Basis construction
* Matrix-free operator application
* Krylov methods
* Parallelization
* Storage
* GPU acceleration
