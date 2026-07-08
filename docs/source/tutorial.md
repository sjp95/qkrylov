# Getting Started with qkrylov

This tutorial will guide you through the basic steps of setting up a quantum many-body simulation using `qkrylov`.

## Library Capabilities and Achievements

The **qkrylov** framework is a high-performance C++20 core with Python bindings designed for research-grade simulations.

### Key Achievements

- **Diverse Physics**: Native support for Spin-Half, spinless Fermion, Hubbard, and t-J models. Support for complex multi-site interactions like 3-spin Dzyaloshinskii–Moriya interaction (DMI).
- **Stable Numerical Solvers**: Lanczos solver with full reorthogonalization for guaranteed convergence. Includes Davidson, Continued Fraction (Dynamics), and Finite Temperature Lanczos Method (FTLM) solvers.
- **Advanced Spectroscopy**: Built-in tools for calculating momentum-resolved higher-order RIXS dynamical susceptibility $S(q, \omega)$.
- **Seamless Integration**: High-performance Python bindings via `nanobind` with a concise, user-friendly syntax.
- **Persistence**: Efficient data storage and retrieval via HDF5 integration.

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
# We want to compute S_x(omega) = -1/pi Im <0 | Sx_0 (omega - H + E0 + i*eta)^-1 Sx_0 | 0 >

# 1. Define the Sx operator
op_sx = qkrylov.OpSum()
op_sx += 0.5, "Sp", 0
op_sx += 0.5, "Sm", 0

# 2. Construct H_sx to apply Sx to the ground state
# (We reuse the basis and site from the Hamiltonian)
H_sx = qkrylov.MatrixFreeHamiltonian(basis, site, op_sx)

# 3. Compute phi0 = Sx | gs >
phi0 = []
H_sx.apply(list(result.eigenvector), phi0)

# 4. Compute Lanczos coefficients for continued fraction
coeffs = qkrylov.continued_fraction_coeffs(H, phi0, n_iter=100)

# 5. Evaluate spectral function at specific frequency omega
omega = 1.5
eta = 0.1
S_w = qkrylov.evaluate_spectral_function(coeffs, omega, result.energy, eta)
print(f"Sx(omega) at {omega} is {S_w}")
```

## 6. Higher-Order RIXS Dynamical Susceptibility

Resonant Inelastic X-ray Scattering (RIXS) involves complex operators and momentum resolution. You can calculate $S_{rixs}(q, \omega)$ by defining a momentum-dependent operator $O(q) = \sum_j e^{iqj} O_j$.

```python
import numpy as np

# Define momentum q
q = np.pi # example for AF momentum
rixs_op_q = qkrylov.OpSum()

for j in range(N):
    phase = np.exp(1j * q * j)
    # Define the local higher-order RIXS operator at site j
    for i in range(j-1, j+2): # small cluster around j
        if i < 0 or i >= N: continue
        # Example: S_j . S_i type interaction in the RIXS operator
        rixs_op_q += phase * 1.0, "Sz", j, "Sz", i
        rixs_op_q += phase * 0.5, "Sp", j, "Sm", i
        rixs_op_q += phase * 0.5, "Sm", j, "Sp", i

# 1. Construct H_q to apply O(q) to the ground state
H_q = qkrylov.MatrixFreeHamiltonian(basis, site, rixs_op_q)

# 2. Compute |phi_q> = O(q) | gs >
phi_q = []
H_q.apply(list(result.eigenvector), phi_q)

# 3. Compute Lanczos coefficients for the continued fraction
coeffs_q = qkrylov.continued_fraction_coeffs(H, phi_q, n_iter=100)

# 4. Evaluate spectral function S_rixs(q, omega)
omega = 1.5
S_q_w = qkrylov.evaluate_spectral_function(coeffs_q, omega, result.energy, eta=0.1)
print(f"S_rixs(q={q}, omega={omega}) = {S_q_w}")
```

## 7. Finite Temperature Simulations

You can use the Finite Temperature Lanczos Method (FTLM) to compute thermodynamic properties.

```python
beta = 1.0 # Inverse temperature 1/T
ftlm_res = qkrylov.ftlm(H, beta, n_random=50, n_steps=100)

print(f"Internal Energy at beta=1: {ftlm_res.internal_energy}")
print(f"Specific Heat at beta=1: {ftlm_res.specific_heat}")
```

## 8. Storing Results with HDF5

Simulation results can be efficiently stored using the `h5py` library in Python.

```python
import h5py
import numpy as np

with h5py.File("result.h5", "w") as f:
    f.create_dataset("energy", data=result.energy)
    f.create_dataset("eigenvector", data=np.array(result.eigenvector))
```

## 9. Parallelization and Acceleration

**qkrylov** supports high-performance parallelization to study larger system sizes.

### Multi-threading (OpenMP)

The library automatically uses OpenMP to parallelize linear algebra operations and Hamiltonian applications. This is enabled by default. To control the number of threads used, set the `OMP_NUM_THREADS` environment variable:

```bash
export OMP_NUM_THREADS=8
python3 my_script.py
```

### GPU Acceleration (CUDA)

CUDA support is available for the Hamiltonian application, which is typically the most expensive part of the calculation.

#### Enabling CUDA

To use CUDA, you must build the library with the `QKRYLOV_ENABLE_CUDA` flag:

```bash
cmake -B build -S . -DQKRYLOV_ENABLE_CUDA=ON
cmake --build build
```

#### Using CUDA from Python

Once built with CUDA support, you can use the `CUDAHamiltonian` class, which has a similar interface to `MatrixFreeHamiltonian`:

```python
# Assuming CUDA support is enabled
H_cuda = qkrylov.CUDAHamiltonian(basis, site, os)
result = qkrylov.lanczos_ground_state(H_cuda)
```

The `CUDAHamiltonian` handles device memory allocation and kernel execution automatically.
