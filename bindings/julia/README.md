# QuantumKrylov.jl: Julia Interface for `qkrylov`

`QuantumKrylov.jl` provides idiomatic Julia bindings for [`qkrylov`](../../README.md), a high-performance C++ library for matrix-free Krylov subspace methods in quantum many-body physics.

The Julia interface is constructed directly on top of the binary-stable C ABI exposed by `libqkrylov.so` via Julia's native `ccall` mechanism. It delivers **zero memory overhead**, **automatic garbage collection**, and **idiomatic mathematical syntax** (e.g., `y = H * x`, `energy, psi = lanczos_ground_state(H; return_state=true)`).

---

## Table of Contents
- [Installation](#installation)
- [Quickstart Example](#quickstart-example)
- [Complete API Reference](#complete-api-reference)
  - [Symmetry Sectors (`Sector`)](#symmetry-sectors-sector)
  - [Site Models (`AbstractSite`)](#site-models-abstractsite)
  - [Hilbert Space Bases (`AbstractBasis`)](#hilbert-space-bases-abstractbasis)
  - [Operator Terms (`OpSum`)](#operator-terms-opsum)
  - [Matrix-Free Hamiltonian (`MatrixFreeHamiltonian`)](#matrix-free-hamiltonian-matrixfreehamiltonian)
  - [Solvers & Dynamics](#solvers--dynamics)
    - [Lanczos Ground State (`lanczos_ground_state`)](#lanczos-ground-state-lanczos_ground_state)
    - [Davidson Eigensolver (`davidson_lowest`)](#davidson-eigensolver-davidson_lowest)
    - [Continued-Fraction Dynamics & Spectral Functions](#continued-fraction-dynamics--spectral-functions)
    - [Finite Temperature Lanczos (`ftlm`)](#finite-temperature-lanczos-ftlm)
- [Multithreading & CPU Core Management](#multithreading--cpu-core-management)
- [Memory Safety & Architecture](#memory-safety--architecture)
- [Running Unit Tests](#running-unit-tests)

---

## Installation

### Option 1: Direct GitHub Installation for latest release (Prebuilt Binaries)

You can install `QuantumKrylov.jl` directly from the `julia-release` branch. Julia's built-in **Artifacts** system automatically downloads and configures the native prebuilt binary (`libqkrylov.so`, `libqkrylov.dylib`, or `qkrylov.dll`) for your operating system and CPU architecture. On Linux systems with an NVIDIA GPU and driver 12+, it automatically downloads the **CUDA 12 accelerated** binary:

In the Julia REPL (press `]` to open Pkg mode):
```julia
pkg> add https://github.com/sjp95/qkrylov.git#julia-release:bindings/julia
```

Or programmatically in Julia scripts:
```julia
using Pkg
Pkg.add(url="https://github.com/sjp95/qkrylov.git", rev="julia-release", subdir="bindings/julia")
```

*(No C++ compiler, CMake, or extra build tools required!)*

### Option 2: Pin to a specific historical build
Every CI build is permanently archived with its own tagged release. To install an exact, pinned build, specify its commit SHA (e.g. `rev="1d00f75"`):
```julia
using Pkg
Pkg.add(url="https://github.com/sjp95/qkrylov.git", rev="<commit-sha>", subdir="bindings/julia")
```

### Option 3: General Registry for stable release (Recommended once merged)
```julia
using Pkg
Pkg.add("QuantumKrylov")
```

### Development Setup (Local Repository)
If you are developing locally from the source repository:
```bash
julia --project=bindings/julia
```

Inside Julia:
```julia
using QuantumKrylov
```

---

## Quickstart Example

Here is a complete example constructing a 4-site spin-1/2 Heisenberg chain, computing its ground state energy and wavefunction, and running the Davidson solver for low-lying excited states (with automatic GPU acceleration when available):

```julia
using QuantumKrylov

# 1. Define Hilbert space basis with Sz = 0 symmetry
sec = Sector()
set_sz!(sec, 0)

basis = SpinHalfBasis(4, sec)

println("Hilbert space dimension: ", dimension(basis)) # Outputs 6

# Inspect basis states
st0 = state(basis, 0) # Bitstring for index 0
println("State index 0 bitstring: ", st0)

# 2. Build 1D Heisenberg model Hamiltonian terms
op = OpSum()
N = 4
for i in 0:(N - 1)
    next_i = mod(i + 1, N)
    # Note: `global` is needed when running as a top-level script, but can be
    # omitted if this loop is inside a function or run directly in the REPL.
    global op += 1.0 * Sz(i) * Sz(next_i) + 0.5 * (Sp(i) * Sm(next_i) + Sm(i) * Sp(next_i))
end

# 3. Create MatrixFreeHamiltonian (site is automatically inferred from basis)
# Automatically targets GPU if available ("cuda:0"), otherwise falls back to CPU
target_dev = is_gpu_build() ? "cuda:0" : "cpu"
H = MatrixFreeHamiltonian(basis, op; device=target_dev)
println("Running on execution device: ", target_dev)

# 4. Perform matrix-vector multiplication (y = H * x)
x = zeros(ComplexF64, dimension(basis))
x[1] = 1.0
y = H * x

# 5. Extract Hamiltonian diagonal elements
diag_H = diagonal(H)
println("Diagonal elements: ", diag_H)

# 6. Compute ground state energy & wavefunction via Lanczos solver
res = lanczos_ground_state(H, maxiter=50, tol=1e-12, return_state=true)
println("Ground State Energy: ", res.energy) # Outputs -2.0
psi = res.state # Vector{ComplexF64} wavefunction

# Destructuring syntax is also supported:
E0, psi_gs = lanczos_ground_state(H, return_state=true)

# 7. Compute lowest 2 eigenvalues & eigenvectors using Davidson solver
dav_res = davidson_lowest(H, n_eig=2, max_subspace=10, tol=1e-6)
println("Lowest 2 Eigenvalues: ", dav_res.eigenvalues)
```

---

## Complete API Reference

### Symmetry Sectors (`Sector`)

Symmetry sectors restrict Hilbert spaces to targeted quantum numbers ($S_z$ projection, particle numbers).

#### `Sector()`
- **Description**: Constructs a new quantum symmetry sector handle.
- **Return**: `Sector` object.

#### `set_sz!(sec::Sector, sz2::Integer)`
- **Description**: Sets the total $S_z$ projection ($2 \times S_z$). For $S_z = 0$, pass `0`. For $S_z = 1/2$, pass `1`.
- **Return**: `sec`

#### `set_hubbard_particles!(sec::Sector, nup::Integer, ndn::Integer)`
- **Description**: Sets electron particle counts for spin-up ($N_{\uparrow}$) and spin-down ($N_{\downarrow}$) sectors in electronic models.
- **Return**: `sec`

#### `set_n!(sec::Sector, n::Integer)`
- **Description**: Sets the total particle number $N$ for spinless fermion models.
- **Return**: `sec`

#### `set_nb!(sec::Sector, nb::Integer)`
- **Description**: Sets the total boson number $N_b$ for bosonic models.
- **Return**: `sec`

---

### Site Models (`AbstractSite`)

Site objects define local degrees of freedom and local operator matrices.

- **`SpinHalfSite()`**: Creates a spin-1/2 site model (dimension 2: $|\uparrow\rangle, |\downarrow\rangle$).
- **`FermionSite()`**: Creates a spinless fermion site model (dimension 2: $|0\rangle, |1\rangle$).
- **`HubbardSite()`**: Creates a spinful Fermi-Hubbard site model (dimension 4: $|0\rangle, |\uparrow\rangle, |\downarrow\rangle, |\uparrow\downarrow\rangle$).
- **`TJSite()`**: Creates a $t$-$J$ model site model with constrained double-occupancy (dimension 3: $|0\rangle, |\uparrow\rangle, |\downarrow\rangle$).

---

### Hilbert Space Bases (`AbstractBasis`)

Basis objects construct quantum many-body state representations across lattice sites.

#### Constructors
- **`SpinHalfBasis(num_sites::Integer, sector=nothing; sz=nothing)`**:
  Constructs a spin-1/2 basis. Accepts optional `sector::Sector` or direct keyword `sz=0` (automatically builds $S_z$ sector).
- **`FermionBasis(num_sites::Integer, sector=nothing; n=nothing)`**:
  Constructs a spinless fermion basis. Accepts optional keyword `n=2` to conserve particle count.
- **`HubbardBasis(num_sites::Integer, sector=nothing; nup=nothing, ndn=nothing)`**:
  Constructs an electronic Fermi-Hubbard basis. Accepts optional keywords `nup=1, ndn=1`.
- **`TJBasis(num_sites::Integer, sector=nothing; nup=nothing, ndn=nothing)`**:
  Constructs a $t$-$J$ model basis with no double-occupancy. Accepts optional keywords `nup=1, ndn=1`.

#### Query & Inspection Methods
- **`dimension(b::AbstractBasis)::UInt64`**: Returns total Hilbert space dimension.
- **`nsites(b::AbstractBasis)::Int`**: Returns physical lattice site count.
- **`state(b::AbstractBasis, index::Integer)::UInt64`**: Returns the basis state bitstring at 0-based `index`.
- **`basis_index(b::AbstractBasis, state_bitstring::Unsigned)::Int64`**: Returns 0-based index of `state_bitstring` (or `-1` if not contained).
- **`Base.in(state_bitstring::Unsigned, b::AbstractBasis)::Bool`**: Checks if `state_bitstring` belongs to basis `b` (`bitstring in basis`).
- **`b[i]`**: 1-based index access returning basis state bitstring at 1-based index `i`.
- **`Base.size(b::AbstractBasis)`**: Returns `(dim, dim)` matrix dimensions.
- **`Base.length(b::AbstractBasis)`**: Returns `dim`.

---

### Operator Terms (`OpSum`)

`OpSum` stores operator term expressions used to construct matrix-free Hamiltonians.

#### Operator Generators & Arithmetic Overloading
Operators support natural mathematical algebra (`+`, `-`, `*`):
- **Spin-1/2**: `Sz(i)`, `Sp(i)`, `Sm(i)`, `Sx(i)`, `Sy(i)`
- **Spinless Fermions**: `c(i)`, `cdag(i)`, `n(i)`
- **Hubbard / Interacting Electrons**: `CdagUp(i)`, `CUp(i)`, `CdagDn(i)`, `CDn(i)`, `Nup(i)`, `Ndn(i)`, `Nupdn(i)`
- **Bosons**: `Bdag(i)`, `B(i)`, `N(i)`

```julia
op = OpSum()
# Expression arithmetic
op += 1.0 * Sz(0) * Sz(1) + 0.5 * (Sp(0) * Sm(1) + Sm(0) * Sp(1))
# Hubbard interaction
op += -1.0 * (CdagUp(0) * CUp(1) + CdagDn(0) * CDn(1)) + 4.0 * Nupdn(0)
```

#### Raw Term Methods
- **`add_term!(op, coeff, op1, site1)`**: Adds 1-body term.
- **`add_term!(op, coeff, op1, site1, op2, site2)`**: Adds 2-body term.
- **`add_term!(op, coeff, ops_vec, sites_vec)`**: Adds arbitrary $N$-body term.
- **`clear!(op)`**: Clears all terms.
- **`validate(op, nsites)`**: Validates site indices against system size.

---

### Matrix-Free Hamiltonian (`MatrixFreeHamiltonian`)

#### Constructors
- **`MatrixFreeHamiltonian(basis, site, opsum; device="cpu")`**:
  Constructs a matrix-free Hamiltonian. Holds reference guards to `basis`, `site`, and `opsum` to guarantee GC safety.
- **`MatrixFreeHamiltonian(basis, opsum; device="cpu")`**:
  Convenience constructor that automatically infers the default `Site` model from `basis`.
- **`device` Keyword**: Specifies execution target (`"cpu"`, `"cuda:0"`, `"hip"`, etc.). Throws an `ArgumentError` if GPU acceleration is requested on a CPU-only build.

#### `dimension(H::MatrixFreeHamiltonian)::UInt64`
- **Description**: Returns the dimension of the Hamiltonian matrix.

#### `diagonal(H::MatrixFreeHamiltonian)::Vector{Float64}`
- **Description**: Extracts diagonal matrix elements $H_{ii}$ into a `Vector{Float64}` without allocating full matrix memory.

#### `Base.:*(H::MatrixFreeHamiltonian, x::AbstractVector{<:Number})::Vector{ComplexF64}`
- **Description**: Computes matrix-vector product $y = H \cdot x$ using zero-copy memory arrays and `GC.@preserve` pointer protection.

---

### Solvers & Dynamics

#### Lanczos Ground State (`lanczos_ground_state`)

```julia
lanczos_ground_state(
    H::MatrixFreeHamiltonian;
    maxiter::Integer=100,
    tol::Real=1e-12,
    return_state::Bool=false
)::LanczosResult
```

- **Arguments**:
  - `H`: Target `MatrixFreeHamiltonian`.
  - `maxiter`: Maximum Lanczos iterations (default: `100`).
  - `tol`: Convergence tolerance for residual norm (default: `1e-12`).
  - `return_state`: If `true`, computes and stores the ground-state wavefunction (default: `false`).
- **Return**: `LanczosResult` struct:
  - `.energy`: Ground state energy (`Float64`).
  - `.state` or `.eigenvector`: Wavefunction vector (`Vector{ComplexF64}`). *Note*: Raises an explicit `ErrorException` if accessed when `return_state=false`.
- **Destructuring**: Supports direct tuple assignment `energy, psi = lanczos_ground_state(H; return_state=true)`.

---

#### Davidson Eigensolver (`davidson_lowest`)

```julia
davidson_lowest(
    H::MatrixFreeHamiltonian;
    n_eig::Integer=1,
    max_subspace::Integer=20,
    tol::Real=1e-6,
    compute_eigenvectors::Bool=true
)::DavidsonResult
```

- **Description**: Computes the lowest $M$ eigenvalues and eigenvectors using subspace expansion.
- **Return**: `DavidsonResult` struct:
  - `.eigenvalues`: `Vector{Float64}` of $M$ lowest eigenvalues.
  - `.eigenvectors`: `Vector{Vector{ComplexF64}}` of $M$ eigenvectors (or `nothing` if `compute_eigenvectors=false`).

---

#### Continued-Fraction Dynamics & Spectral Functions

```julia
continued_fraction_coeffs(
    H::MatrixFreeHamiltonian,
    phi0::AbstractVector{<:Number};
    n_iter::Integer=100
)::ContinuedFractionResult
```
- **Description**: Computes Lanczos tridiagonal coefficients ($\alpha_n, \beta_n$) starting from state vector `phi0`.
- **Return**: `ContinuedFractionResult` struct containing `.alphas`, `.betas`, and `.norm_phi0`.

```julia
evaluate_spectral_function(
    cfr::ContinuedFractionResult,
    omega::Real,
    E0::Real,
    eta::Real
)::Float64
```
- **Description**: Evaluates the spectral function $I(\omega) = -\frac{1}{\pi} \text{Im} \langle \phi_0 | \frac{1}{\omega + E_0 + i\eta - H} | \phi_0 \rangle$ at frequency `omega` with broadening `eta` and ground state energy `E0`.

---

#### Finite Temperature Lanczos (`ftlm`)

```julia
ftlm(
    H::MatrixFreeHamiltonian;
    beta::Real=1.0,
    n_random::Integer=10,
    n_steps::Integer=50
)::FTLMResult
```

- **Description**: Calculates thermodynamic properties at inverse temperature $\beta = 1 / k_B T$ using random sampling vectors.
- **Return**: `FTLMResult` struct containing:
  - `.beta`: Inverse temperature $\beta$.
  - `.partition_function`: Thermal partition function $Z(\beta)$.
  - `.internal_energy`: Internal energy $E(\beta)$.
  - `.specific_heat`: Specific heat capacity $C_v(\beta)$.

---

## Multithreading & CPU Core Management

`QuantumKrylov.jl` uses a high-performance C++ backend powered by **Kokkos with OpenMP** for multi-core CPU parallelism (matrix-free applies $y = H \cdot x$, linear algebra, and solver iterations).

### Understanding Julia Threads vs. OpenMP Threads

Julia task threads and C++ OpenMP threads are **completely independent**:

| Thread Pool | Configured Via | Governs |
| :--- | :--- | :--- |
| **Julia Runtime Threads** | `julia -t N` or `JULIA_NUM_THREADS` | Julia-level concurrency (`Threads.@threads`, `Threads.@spawn`) |
| **OpenMP C++ Threads** | `OMP_NUM_THREADS=M` | `qkrylov` C++ backend parallelism, matrix-vector products, solvers |

*Note: Starting Julia with `julia -t 1` only limits Julia's internal tasks. By default, OpenMP will still detect and utilize all available CPU cores unless `OMP_NUM_THREADS` is set.*

### How to Control C++ Worker Threads

#### 1. From the Terminal (Recommended)
```bash
# Force both Julia and qkrylov to use 1 thread:
OMP_NUM_THREADS=1 julia -t 1

# Limit qkrylov C++ parallel kernels to 4 cores:
OMP_NUM_THREADS=4 julia -t 1
```

#### 2. Inside Julia (Before Loading the Package)
```julia
# Set OpenMP thread count before initializing QuantumKrylov
ENV["OMP_NUM_THREADS"] = "4"
using QuantumKrylov
```

### Best Practices for High Performance

- **Single Large System**: Use `julia -t 1` and let `OMP_NUM_THREADS` use all available physical CPU cores for maximum parallel solver throughput.
- **Julia Parallel Parameter Sweeps (`Threads.@threads`)**: Set `OMP_NUM_THREADS=1` and `julia -t N` to prevent CPU oversubscription from nested thread pools.
- **Recommended OpenMP Affinity**: For best cache locality and NUMA performance:
  ```bash
  export OMP_PROC_BIND=spread
  export OMP_PLACES=threads
  ```

### Device & Hardware Query API

Query compiled accelerator capabilities and initialize execution targets:

```julia
using QuantumKrylov

# Check if qkrylov was built with GPU acceleration (CUDA, HIP, SYCL)
is_gpu = is_gpu_build()  # Returns Bool

# Query active GPU backend name ("cuda", "hip", "sycl", or nothing)
gpu = find_gpu()

# Query number of physical GPUs detected on system
count = gpu_count()      # Returns Int

# Explicitly initialize device runtime (e.g. "cpu", "cuda:0")
initialize_device!("cpu")
```

---

## Memory Safety & Architecture

- **Automatic Garbage Collection**: Every Julia struct (`Sector`, `AbstractSite`, `AbstractBasis`, `OpSum`, `MatrixFreeHamiltonian`) registers a Julia `finalizer` block on creation. When Julia garbage collects an object, the corresponding C ABI destructor function (`qkrylov_*_destroy`) is automatically called.
- **Reference Preservation**: `MatrixFreeHamiltonian` stores fields pointing to its `basis`, `site`, and `opsum`. This guarantees that Julia will never garbage collect dependency handles while the Hamiltonian object exists.
- **Zero-Copy Matrix-Vector Application**: Vector operations use `GC.@preserve` during `ccall` invocations, passing raw pointers to Julia array memory directly into C++ core kernels without copying data.

---

## Running Unit Tests

Run the full package test suite using Julia:
```bash
julia --project=bindings/julia -e 'using Pkg; Pkg.test()'
```
