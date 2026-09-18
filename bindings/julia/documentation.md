# QuantumKrylov.jl Documentation: Core Concepts

`QuantumKrylov.jl` provides three fundamental building blocks for setting up quantum many-body systems: **Sectors**, **Sites**, and **Bases**.

---

## 1. Symmetry Sectors (`Sector`)

### What is a Sector?
A `Sector` represents quantum conservation laws and symmetry restrictions (such as total spin $S_z$, spin-up/spin-down particle numbers $N_\uparrow, N_\downarrow$, total fermion count $N$, or boson count $N_b$).

Using a symmetry sector restricts the Hilbert space to states matching specific quantum numbers, significantly reducing memory consumption and matrix dimensions.

### How to use `Sector`

```julia
using QuantumKrylov

# Create an empty sector object (no constraints enabled initially)
sec = Sector()
println(sec) # Outputs: Sector(unconstrained)

# Constrain total Sz projection (2 * Sz)
set_sz!(sec, 0) # Sz = 0
println(sec) # Outputs: Sector(2*Sz = 0)

# Check active constraints
sz_val = get_sz(sec) # Returns 0 (Int)
n_val  = get_n(sec)  # Returns nothing (unconstrained)

# Constrain electron numbers for Fermi-Hubbard models
set_hubbard_particles!(sec, 1, 1) # N_up = 1, N_down = 1
hp_val = get_hubbard_particles(sec) # Returns (1, 1)

# Constrain spinless fermion particle count
set_n!(sec, 2) # N = 2 particles
get_n(sec)     # Returns 2

# Constrain boson particle count
set_nb!(sec, 1) # Nb = 1 boson
get_nb(sec)     # Returns 1
```

### Functions & Default Values

| Function | Parameters | Default Values | Description |
| :--- | :--- | :--- | :--- |
| `Sector()` | None | Unconstrained sector handle | Constructs a new symmetry sector with no quantum number constraints active. |
| `set_sz!(sec, sz2)` | `sec::Sector`<br>`sz2::Integer` | None (required) | Restricts basis to states with total $S_z = \text{sz2} / 2$. Returns `sec`. |
| `set_hubbard_particles!(sec, nup, ndn)` | `sec::Sector`<br>`nup::Integer`<br>`ndn::Integer` | None (required) | Restricts basis to states with $N_\uparrow = \text{nup}$ and $N_\downarrow = \text{ndn}$. Returns `sec`. |
| `set_n!(sec, n)` | `sec::Sector`<br>`n::Integer` | None (required) | Restricts basis to states with $N = \text{n}$ total spinless fermions. Returns `sec`. |
| `set_nb!(sec, nb)` | `sec::Sector`<br>`nb::Integer` | None (required) | Restricts basis to states with $N_b = \text{nb}$ total bosons. Returns `sec`. |
| `get_sz(sec)` | `sec::Sector` | None | Returns active $2 S_z$ value (`Int`), or `nothing` if unconstrained. |
| `get_hubbard_particles(sec)` | `sec::Sector` | None | Returns `(nup, ndn)` tuple (`Tuple{Int, Int}`), or `nothing` if unconstrained. |
| `get_n(sec)` | `sec::Sector` | None | Returns active $N$ particle count (`Int`), or `nothing` if unconstrained. |
| `get_nb(sec)` | `sec::Sector` | None | Returns active $N_b$ boson count (`Int`), or `nothing` if unconstrained. |

---

## 2. Site Models (`AbstractSite`)

### What is a Site?
A `Site` object defines the physical degree of freedom at each lattice site, specifying the local Hilbert space dimension and local quantum operators (such as $S^z, S^+, S^-, c, c^\dagger, n$).

### How to use `Site`

```julia
# Spin-1/2 local degree of freedom (dimension 2)
site_spin = SpinHalfSite()
println(site_spin) # Outputs: SpinHalfSite(dim = 2, states = [↑, ↓])

# Spinless fermion local degree of freedom (dimension 2)
site_fermion = FermionSite()
println(site_fermion) # Outputs: FermionSite(dim = 2, states = [0, 1])

# Spinful Fermi-Hubbard electron site (dimension 4)
site_hubbard = HubbardSite()
println(site_hubbard) # Outputs: HubbardSite(dim = 4, states = [0, ↑, ↓, ↑↓])

# General Spin-S site (e.g. S=1, dimension 2S+1 = 3)
site_spin1 = SpinSSite(1.0)
println(site_spin1) # Outputs: SpinSSite(S = 1.0, dim = 3)

# t-J model site with constrained double-occupancy (dimension 3)
site_tj = TJSite()
println(site_tj) # Outputs: TJSite(dim = 3, states = [0, ↑, ↓])
```

### Available Site Types & Default Values

| Site Type | Local Dimension | Local Physical Basis States | Arguments & Default Values |
| :--- | :--- | :--- | :--- |
| `SpinHalfSite()` | 2 | $|\uparrow\rangle, |\downarrow\rangle$ | No arguments required. |
| `SpinSSite(S)` | $2S + 1$ | $|S, m\rangle$ for $m = -S, \dots, +S$ | `S::Real` (e.g. `0.5`, `1.0`, `1.5`, `2.0`). |
| `FermionSite()` | 2 | $|0\rangle, |1\rangle$ (empty, occupied) | No arguments required. |
| `HubbardSite()` | 4 | $|0\rangle, |\uparrow\rangle, |\downarrow\rangle, |\uparrow\downarrow\rangle$ | No arguments required. |
| `TJSite()` | 3 | $|0\rangle, |\uparrow\rangle, |\downarrow\rangle$ (no double occupancy) | No arguments required. |

---

## 3. Hilbert Space Bases (`AbstractBasis`)

### What is a Basis?
A `Basis` represents the complete many-body Hilbert space constructed across $N$ physical lattice sites. It can either represent the full unconstrained Hilbert space or be restricted to a specific `Sector`.

### How to use `Basis`

```julia
# 1. Construct a full (unconstrained) 4-site spin-1/2 basis
basis_full = SpinHalfBasis(4)
println(basis_full) # Outputs: SpinHalfBasis(sites = 4, dim = 16)

# 2. Construct a 4-site spin-1/2 basis restricted to Sz = 0 sector
sec = Sector()
set_sz!(sec, 0)
basis_sec = SpinHalfBasis(4, sec)
println(basis_sec) # Outputs: SpinHalfBasis(sites = 4, dim = 6, sector = Sector(2*Sz = 0))

# 3. Query properties
N = nsites(basis_sec) # 4
dim = dimension(basis_sec) # 6

# 4. Inspect basis state bitstrings
st0 = state(basis_sec, 0) # Bitstring of 0-th state (0-indexed)
st_first = basis_sec[1]   # Bitstring of 1st state (1-indexed Julia syntax)

# 5. Look up index from bitstring
idx = basis_index(basis_sec, st0) # Returns 0

# 6. Check if state bitstring belongs to basis
is_present = st0 in basis_sec # Returns true
```

### Basis Types, Parameters & Default Values

| Basis Constructor | Parameters | Default Values | Description |
| :--- | :--- | :--- | :--- |
| `SpinHalfBasis(num_sites, sector=nothing; sz=nothing)` | `num_sites::Integer`<br>`sector::Union{Sector, Nothing}`<br>`sz::Union{Real, Nothing}` | `num_sites`: Required<br>`sector`: `nothing`<br>`sz`: `nothing` | Spin-1/2 basis on `num_sites` sites. Passing `sz=0` automatically builds total $S_z=0$ sector. |
| `SpinSBasis(num_sites, S; sector=nothing)` | `num_sites::Integer`<br>`S::Real`<br>`sector::Union{Sector, Nothing}` | `num_sites`: Required<br>`S`: Required<br>`sector`: `nothing` | General spin-$S$ basis (e.g. $S=1.0, 1.5, \dots$) on `num_sites` sites with optional $S_z$ sector conservation. |
| `FermionBasis(num_sites, sector=nothing; n=nothing)` | `num_sites::Integer`<br>`sector::Union{Sector, Nothing}`<br>`n::Union{Integer, Nothing}` | `num_sites`: Required<br>`sector`: `nothing`<br>`n`: `nothing` | Spinless fermion basis on `num_sites` sites. Passing `n=2` restricts to 2-particle sector. |
| `HubbardBasis(num_sites, sector=nothing; nup=nothing, ndn=nothing)` | `num_sites::Integer`<br>`sector::Union{Sector, Nothing}`<br>`nup, ndn::Union{Integer, Nothing}` | `num_sites`: Required<br>`sector`: `nothing`<br>`nup, ndn`: `nothing` | Fermi-Hubbard basis on `num_sites` sites with optional electron number conservation. |
| `TJBasis(num_sites, sector=nothing; nup=nothing, ndn=nothing)` | `num_sites::Integer`<br>`sector::Union{Sector, Nothing}`<br>`nup, ndn::Union{Integer, Nothing}` | `num_sites`: Required<br>`sector`: `nothing`<br>`nup, ndn`: `nothing` | $t$-$J$ model basis on `num_sites` sites with optional electron number conservation. |

### Basis Operations & Query Functions

| Function / Syntax | Arguments | Return Type | Description |
| :--- | :--- | :--- | :--- |
| `dimension(b)` | `b::AbstractBasis` | `UInt64` | Returns total dimension of the basis space. |
| `nsites(b)` | `b::AbstractBasis` | `Int` | Returns number of physical lattice sites. |
| `state(b, index)` | `b::AbstractBasis`, `index::Integer` | `UInt64` | Returns 64-bit integer bitstring of state at 0-based `index`. |
| `b[i]` | `b::AbstractBasis`, `i::Integer` | `UInt64` | Returns 64-bit integer bitstring of state at 1-based index `i`. |
| `basis_index(b, bitstring)`| `b::AbstractBasis`, `bitstring::Unsigned` | `Int64` | Returns 0-based index of `bitstring` in basis, or `-1` if not present. |
| `bitstring in b` | `bitstring::Unsigned`, `b::AbstractBasis` | `Bool` | Returns `true` if `bitstring` state belongs to basis `b`. |
| `size(b)` | `b::AbstractBasis` | `(Int, Int)` | Returns `(dim, dim)` tuple. |
| `length(b)` | `b::AbstractBasis` | `Int` | Returns total dimension `dim`. |

---

## 4. Operator Terms (`OpSum`) & Matrix-Free Hamiltonians (`MatrixFreeHamiltonian`)

### 4.1 Operator Terms (`OpSum`)

#### What is an `OpSum`?
`OpSum` stores operator term expressions (such as $\hat{S}_i^z \hat{S}_j^z$, $\hat{S}_i^+ \hat{S}_j^-$, or $c_i^\dagger c_j$) used to construct a Hamiltonian or observable operator. Terms are built using local operator string names (e.g. `"Sz"`, `"Sp"`, `"Sm"`, `"n"`, `"c"`, `"cdag"`) and 0-indexed site numbers.

#### How to use `OpSum`

```julia
# 1. Create an empty OpSum
op = OpSum()

# 2. Operator Arithmetic Syntax (Recommended)
# Build 1D Heisenberg model terms directly using operator generators:
for i in 0:3
    next_i = mod(i + 1, 4)
    op += 1.0 * Sz(i) * Sz(next_i) + 0.5 * (Sp(i) * Sm(next_i) + Sm(i) * Sp(next_i))
end

# 3. Add a 3-body (N-body) term using operator generators
op += 0.25 * Sz(0) * Sz(1) * Sz(2)

# 4. Inspect OpSum
println(op)
# Outputs: OpSum(Sz(0) * Sz(1) + 0.5 * Sp(0) * Sm(1) + 0.5 * Sm(0) * Sp(1) + ...)

num_terms = length(op) # Returns 5
is_empty  = isempty(op) # Returns false

# 5. Validate site index bounds for a 4-site system
valid, errors = validate(op, 4) # Returns (true, String[])
validate!(op, 4)                # Throws ArgumentError if any site >= 4 or < 0

# 6. Alternatively, use string-based add_term! functions:
add_term!(op, 1.0, "Sz", 0, "Sz", 1)

# 7. Clear all terms
clear!(op)
```

#### Operator Generators & Arithmetic Overloads

| Generator / Syntax | Arguments | Description |
| :--- | :--- | :--- |
| `Sz(site)`, `Sp(site)`, `Sm(site)` | `site::Integer` | Spin-1/2 operators ($S^z, S^+, S^-$) at 0-indexed `site`. |
| `Sx(site)`, `Sy(site)` | `site::Integer` | Spin-1/2 operators ($S^x, S^y$) at 0-indexed `site`. |
| `n(site)` | `site::Integer` | Particle number operator ($n_i$) at 0-indexed `site`. |
| `c(site)`, `cdag(site)` | `site::Integer` | Fermionic annihilation ($c_i$) and creation ($c_i^\dagger$) operators at 0-indexed `site`. |
| `CdagUp(site)`, `CUp(site)` | `site::Integer` | Spin-up electron creation ($c_{i,\uparrow}^\dagger$) and annihilation ($c_{i,\uparrow}$) operators. |
| `CdagDn(site)`, `CDn(site)` | `site::Integer` | Spin-down electron creation ($c_{i,\downarrow}^\dagger$) and annihilation ($c_{i,\downarrow}$) operators. |
| `Nup(site)`, `Ndn(site)`, `Nupdn(site)` | `site::Integer` | Electron number operators ($n_{i,\uparrow}, n_{i,\downarrow}, n_{i,\uparrow} n_{i,\downarrow}$). |
| `Bdag(site)`, `B(site)`, `N(site)` | `site::Integer` | Boson creation ($b_i^\dagger$), annihilation ($b_i$), and number ($n_i$) operators. |
| `coeff * term * ...` | `coeff::Number`, `term::OpTerm` | Multiplies operator factors and scales coupling coefficient. |
| `term1 + term2` | `term1`, `term2` | Combines operator terms into an `OpExpr` term collection. |
| `op += expr` | `op::OpSum`, `expr::OpExpr` | Appends operator terms into `OpSum`. |

#### `OpSum` Functions & Default Values

| Function | Parameters | Default Values | Description |
| :--- | :--- | :--- | :--- |
| `OpSum()` | None | Empty `OpSum` handle | Constructs a new empty operator sum container. |
| `length(op)` | `op::OpSum` | None | Returns the total number of terms stored in `op`. |
| `isempty(op)` | `op::OpSum` | None | Returns `true` if `op` has zero terms. |
| `validate(op, num_sites)` | `op::OpSum`<br>`num_sites::Integer` | None (required) | Validates term site indices against `0 <= site < num_sites`. Returns `(valid::Bool, errors::Vector{String})`. |
| `validate!(op, num_sites)` | `op::OpSum`<br>`num_sites::Integer` | None (required) | Validates term site indices. Throws `ArgumentError` if any site index is out of bounds or coefficient is non-finite. |
| `add_term!(op, coeff, op1, site1)` | `op::OpSum`<br>`coeff::Number`<br>`op1::AbstractString`<br>`site1::Integer` | None (required) | Adds 1-body term $\text{coeff} \cdot \hat{O}_{1, \text{site1}}$. Returns `op`. |
| `add_term!(op, coeff, op1, site1, op2, site2)` | `op::OpSum`<br>`coeff::Number`<br>`op1::AbstractString`<br>`site1::Integer`<br>`op2::AbstractString`<br>`site2::Integer` | None (required) | Adds 2-body term $\text{coeff} \cdot \hat{O}_{1, \text{site1}} \hat{O}_{2, \text{site2}}$. Returns `op`. |
| `add_term!(op, coeff, ops, sites)` | `op::OpSum`<br>`coeff::Number`<br>`ops::Vector{<:AbstractString}`<br>`sites::Vector{<:Integer}` | None (required) | Adds general $N$-body term $\text{coeff} \cdot \prod_{k=1}^N \hat{O}_{k, \text{sites}[k]}$. Returns `op`. |
| `clear!(op)` | `op::OpSum` | None | Clears all added operator terms from `op`. Returns `op`. |

---

### 4.2 Matrix-Free Hamiltonian (`MatrixFreeHamiltonian`)

#### What is a `MatrixFreeHamiltonian`?
A `MatrixFreeHamiltonian` evaluates matrix-vector products $y = H \cdot x$ on-the-fly without ever storing the full $N \times N$ matrix in memory. It combines a `Basis`, a `Site` model, and an `OpSum`.

#### How to use `MatrixFreeHamiltonian`

```julia
# 1. Setup basis and opsum
basis = SpinHalfBasis(4)
op    = OpSum()
add_term!(op, 1.0, "Sz", 0, "Sz", 1)

# 2. Construct MatrixFreeHamiltonian (site is automatically inferred from basis)
# Automatically targets GPU if available ("cuda:0", "cuda"), otherwise falls back to CPU
target_dev = is_gpu_build() ? "cuda:0" : "cpu"
H = MatrixFreeHamiltonian(basis, op; device=target_dev)
println(H) # Outputs: MatrixFreeHamiltonian(dim = 16, basis = SpinHalfBasis(sites = 4, dim = 16))

# Alternatively, explicitly specify the site model and execution target:
site = SpinHalfSite()
H_explicit = MatrixFreeHamiltonian(basis, site, op; device="cpu")

# 3. Perform matrix-vector multiplication (y = H * x)
# Evaluated on the target device (Kokkos OpenMP on CPU, or CUDA on GPU)
x = zeros(ComplexF64, dimension(H))
x[1] = 1.0 + 0.0im
y = H * x # Vector{ComplexF64} of length dimension(H)

# 4. Extract diagonal elements H_ii without matrix allocation
diag_H = diagonal(H) # Vector{Float64} of length dimension(H)

# 5. Query dimensions
dim = dimension(H) # 16
sz  = size(H)      # (16, 16)
```

#### `MatrixFreeHamiltonian` Functions & Operations

| Function / Syntax | Arguments | Return Type | Description |
| :--- | :--- | :--- | :--- |
| `MatrixFreeHamiltonian(basis, opsum; device="cpu")` | `basis::AbstractBasis`<br>`opsum::OpSum`<br>`device::AbstractString="cpu"` | `MatrixFreeHamiltonian` | **Convenience Constructor**. Automatically infers the matching default `Site` model (`SpinHalfSite`, `FermionSite`, `HubbardSite`, or `TJSite`) from the basis type. Targets specified execution `device` (`"cpu"`, `"cuda:0"`, `"hip"`, etc.). |
| `MatrixFreeHamiltonian(basis, site, opsum; device="cpu")` | `basis::AbstractBasis`<br>`site::AbstractSite`<br>`opsum::OpSum`<br>`device::AbstractString="cpu"` | `MatrixFreeHamiltonian` | **Explicit Constructor**. Constructs a matrix-free Hamiltonian operator with a specified site model on target `device`. |
| `H * x` | `H::MatrixFreeHamiltonian`<br>`x::AbstractVector{<:Number}` | `Vector{ComplexF64}` | Performs zero-copy matrix-vector multiplication $y = H \cdot x$ on target device. Length of `x` must equal `dimension(H)`. |
| `diagonal(H)` | `H::MatrixFreeHamiltonian` | `Vector{Float64}` | Computes and returns matrix-free diagonal elements $H_{ii}$. |
| `dimension(H)` | `H::MatrixFreeHamiltonian` | `UInt64` | Returns total matrix dimension of `H`. |
| `size(H)` | `H::MatrixFreeHamiltonian` | `(Int, Int)` | Returns `(dim, dim)` matrix shape tuple. |

---

## 5. Krylov & Lanczos Solvers

> **Hardware Acceleration (CPU & GPU)**:
> All solvers below inherit the execution space of the supplied `MatrixFreeHamiltonian`. If `H` was created with `device="cuda:0"`, all matrix-vector multiplications ($y = H \cdot x$), Krylov projection steps, and vector updates execute entirely on the GPU via Kokkos CUDA kernels without host-device transfer bottlenecks.

### 5.1 Ground State Solver (`lanczos_ground_state`)

#### How to use `lanczos_ground_state`

```julia
# 1. Energy-only calculation (runs on H's target device: CPU or GPU)
res = lanczos_ground_state(H, maxiter=200, tol=1e-12)
println(res)            # Outputs: LanczosResult(energy = -2.0, iterations = 14, converged = true)
E0 = res.energy          # Ground state energy Float64
n_iters = res.iterations # Number of Lanczos iterations executed (14)
is_conv = res.converged  # true (false if maxiter was hit without meeting tol)

# If maxiter is reached before achieving tolerance `tol`:
# println(res) -> LanczosResult(energy = -1.987, iterations = 5, WARNING: maxiter hit without converging!)

# 2. Compute Ground State Wavefunction (|psi_0>)
res = lanczos_ground_state(H, return_state=true)
println(res)            # Outputs: LanczosResult(energy = -2.0, iterations = 14, converged = true, state = Vector{ComplexF64}(dim=6))
psi0 = res.state         # Vector{ComplexF64} of length dimension(H)
psi0_vec = res.eigenvector # Alternative alias for wavefunction

# 3. Memory-Frugal Two-Pass Lanczos (saves only 3 working vectors in RAM)
res_tp = lanczos_ground_state(H, return_state=true, two_pass=true)

# 4. Tuple Destructuring Support
E0, psi0 = lanczos_ground_state(H, return_state=true)
```

#### `lanczos_ground_state` Parameters & Result API

| Parameter / Property | Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `maxiter` | `Integer` | `100` | Maximum Lanczos iterations. |
| `tol` | `Real` | `1e-12` | Energy convergence tolerance. |
| `return_state` | `Bool` | `false` | When `true`, computes and stores ground state wavefunction. |
| `two_pass` | `Bool` | `false` | When `true`, uses a memory-frugal two-pass Lanczos algorithm that requires only 3 working vectors in RAM, reconstructing the Ritz eigenvector on-the-fly in a second pass. |
| `res.energy` | `Float64` | - | Ground state energy eigenvalue. |
| `res.iterations` | `Int` | - | Total number of Lanczos iterations executed. |
| `res.converged` | `Bool` | - | `true` if tolerance `tol` was achieved, `false` if `maxiter` was hit. |
| `res.state` / `res.eigenvector` | `Vector{ComplexF64}` | - | Ground state eigenvector (raises Error if `return_state=false`). |

---

### 5.2 Low-Lying Excited States (`davidson_lowest`)

#### How to use `davidson_lowest`

```julia
# Compute lowest 3 eigenvalues and eigenvectors using Davidson solver
res_dav = davidson_lowest(H, n_eig=3, max_subspace=30, tol=1e-8)
println(res_dav)
# Outputs: DavidsonResult(n_eig = 3, energies = [-2.0, -1.0, -1.0], iterations = 8, converged = true, has_eigenvectors = true)

energies = res_dav.eigenvalues   # Vector{Float64} of length 3
states   = res_dav.eigenvectors  # Vector{Vector{ComplexF64}} of length 3
n_iters  = res_dav.iterations    # Iteration count
is_conv  = res_dav.converged     # Convergence boolean flag
```

#### `davidson_lowest` Parameters & Result API

| Parameter / Property | Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `n_eig` | `Integer` | `1` | Number of lowest eigenvalues to compute. |
| `max_subspace` | `Integer` | `20` | Maximum Davidson search subspace dimension. |
| `tol` | `Real` | `1e-6` | Convergence tolerance. |
| `compute_eigenvectors` | `Bool` | `true` | When `true`, computes eigenvector array. |
| `res.eigenvalues` | `Vector{Float64}` | - | Vector of $k$ lowest energy eigenvalues. |
| `res.iterations` | `Int` | - | Number of Davidson subspace iterations executed. |
| `res.converged` | `Bool` | - | `true` if all $k$ eigenpairs converged within tolerance (`false` if `maxiter` reached). |
| `res.eigenvectors` | `Vector{Vector{ComplexF64}}` | - | Array of $k$ eigenvector state vectors. |

---

### 5.3 Dynamical Response & Spectral Functions (`continued_fraction_coeffs`)

#### How to use Continued Fraction Dynamics

```julia
# 1. Prepare excited state v0 = A * |psi0>
v0 = H * psi0

# 2. Compute continued-fraction Lanczos coefficients alpha_k, beta_k
cfr = continued_fraction_coeffs(H, v0, n_iter=100)
println(cfr) # Outputs: ContinuedFractionResult(n_coeffs = 100, norm_phi0 = 1.0)

# 3. Evaluate dynamical spectral function I(omega) at energy omega
E0 = res.energy
eta = 0.05 # Lorentzian broadening
omega = 0.5
I_omega = evaluate_spectral_function(cfr, omega, E0, eta)
```

#### Dynamics Functions & Parameters

| Function | Parameters | Default Values | Description |
| :--- | :--- | :--- | :--- |
| `continued_fraction_coeffs(H, phi0)` | `H::MatrixFreeHamiltonian`<br>`phi0::AbstractVector`<br>`n_iter::Integer` | `n_iter=100` | Computes tridiagonal Lanczos coefficients $\alpha_k, \beta_k$. Returns `ContinuedFractionResult`. |
| `evaluate_spectral_function(cfr, omega, E0, eta)` | `cfr::ContinuedFractionResult`<br>`omega::Real`<br>`E0::Real`<br>`eta::Real` | None (required) | Evaluates spectral response function $I(\omega) = -\frac{1}{\pi} \text{Im} G(\omega + E_0 + i\eta)$. |

---

### 5.4 Correction Vector Method (`solver_correction_vector`)

The **correction vector method** directly solves the complex shifted linear system:

$$(H - E_0 - \omega - i\eta) |x\rangle = \hat{O} |\psi_0\rangle$$

providing high-resolution spectral response at target frequency $\omega$ without Krylov truncation error.

#### How to use `solver_correction_vector`

```julia
# Solve linear system for target frequency omega = 1.0
cv_res = solver_correction_vector(
    H, v0;
    e0 = E0,
    omega = 1.0,
    eta = 0.05,
    maxiter = 500,
    tol = 1e-8,
    return_vector = true
)

println("Spectral weight: ", cv_res.spectral_function)
println("Iterations:      ", cv_res.iterations)
println("Converged:       ", cv_res.converged)
x_vec = cv_res.vector # Vector{ComplexF64} correction vector
```

#### Parameters & Result API

| Parameter / Property | Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `e0` | `Real` | None (required) | Ground-state reference energy $E_0$. |
| `omega` | `Real` | None (required) | Target excitation frequency $\omega$. |
| `eta` | `Real` | `0.1` | Imaginary broadening parameter $\eta > 0$. |
| `maxiter` | `Integer` | `500` | Maximum linear solver iterations. |
| `tol` | `Real` | `1e-8` | Residual convergence tolerance. |
| `return_vector` | `Bool` | `false` | When `true`, returns the correction vector state. |
| `res.spectral_function` | `Float64` | - | Spectral weight $I(\omega) = \frac{\eta}{\pi} \langle x | x \rangle$. |
| `res.iterations` | `Int` | - | Number of linear solver iterations executed. |
| `res.converged` | `Bool` | - | `true` if residual tolerance was satisfied. |
| `res.vector` | `Vector{ComplexF64}` | - | Correction vector state (error if `return_vector=false`). |

---

### 5.5 Finite Temperature Lanczos (`ftlm` & `ftlm_sweep`)

`qkrylov` supports both single-temperature calculations and decoupled multi-temperature sweeps with arbitrary physical observables.

#### 1. Single Temperature (`ftlm`)

```julia
ftlm_res = ftlm(H, beta=2.0, n_random=20, n_steps=50)
println("Partition function Z: ", ftlm_res.partition_function)
println("Internal energy E:    ", ftlm_res.internal_energy)
println("Specific heat Cv:     ", ftlm_res.specific_heat)
```

#### 2. Multi-Temperature Sweeps & Observables (`ftlm_sweep`)

Performs Krylov sampling once, projecting arbitrary user observables $\hat{O}$ onto the Krylov subspace. Thermodynamic quantities and observables are then evaluated across the entire $\beta$ grid in a single pass:

```julia
# Temperature grid and observable operators
betas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
sweep_res = ftlm_sweep(
    H, betas;
    observables = [H], # Arbitrary MatrixFreeHamiltonian operators
    n_random = 20,
    n_steps = 50,
    seed = 42
)

println("Free energies:    ", sweep_res.free_energies)
println("Specific heats:   ", sweep_res.specific_heats)
println("Entropies:        ", sweep_res.entropies)
println("Energy <H>:       ", sweep_res.observable_expectations[1, :])
println("Sampling errors:  ", sweep_res.observable_errors[1, :])
```

#### `ftlm_sweep` Parameters & Result API

| Parameter / Property | Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `beta_grid` | `AbstractVector{<:Real}` | None (required) | Grid of inverse temperatures $\beta = 1 / (k_B T)$. |
| `observables` | `Vector{<:MatrixFreeHamiltonian}` | `[]` | List of observable operators to measure. |
| `n_random` | `Integer` | `10` | Number of random sampling vectors. |
| `n_steps` | `Integer` | `50` | Lanczos expansion steps per sample. |
| `seed` | `Integer` | `0` | PRNG seed (0 for non-deterministic). |
| `res.beta_grid` | `Vector{Float64}` | - | Inverse temperature grid. |
| `res.partition_functions` | `Vector{Float64}` | - | Partition functions $Z(\beta)$. |
| `res.free_energies` | `Vector{Float64}` | - | Helmholtz free energies $F(\beta) = -\frac{1}{\beta} \ln Z$. |
| `res.internal_energies` | `Vector{Float64}` | - | Internal energies $\langle E \rangle_\beta$. |
| `res.specific_heats` | `Vector{Float64}` | - | Specific heats $C_v(\beta) = \beta^2 (\langle E^2 \rangle - \langle E \rangle^2)$. |
| `res.entropies` | `Vector{Float64}` | - | Thermal entropies $S(\beta) = \beta (E - F)$. |
| `res.observable_expectations` | `Matrix{Float64}` | - | Matrix of shape `(n_obs, n_betas)` of expectation values. |
| `res.observable_errors` | `Matrix{Float64}` | - | Matrix of shape `(n_obs, n_betas)` of sampling standard errors. |

---

### 5.6 SciML CommonSolve Interface (`solve(prob, alg)`)

`QuantumKrylov.jl` implements standard problem and algorithm dispatches following the Julia SciML `CommonSolve` pattern:

```julia
# 1. Ground state calculation
prob_gs = GroundStateProblem(H)
sol_gs  = solve(prob_gs, Lanczos(maxiter=100, tol=1e-12, compute_eigenvector=true))

# 2. Excited states calculation
prob_es = ExcitedStatesProblem(H, 3)
sol_es  = solve(prob_es, Davidson(n_eig=3, max_subspace=20, tol=1e-8))

# 3. Thermal sweeps
prob_th = ThermalProblem(H, [0.1, 0.5, 1.0, 2.0]; observables=[H])
sol_th  = solve(prob_th, FTLM(n_random=20, n_steps=50))

# 4. Dynamical response
prob_dyn = DynamicalProblem(H, v0; e0=E0)
sol_dyn  = solve(prob_dyn, ContinuedFraction(n_iter=100))

# 5. Correction vector shifted linear solve
prob_cv = CorrectionVectorProblem(H, v0; e0=E0, omega=1.0, eta=0.05)
sol_cv  = solve(prob_cv, CorrectionVector(maxiter=500, tol=1e-8, return_vector=true))
```

---

## 6. Multithreading & Hardware Resource Control

`QuantumKrylov.jl` wraps a high-performance C++ engine accelerated by **Kokkos OpenMP** parallelism.

### 6.1 Julia Runtime Threads vs. OpenMP Backend Threads

Julia thread management and native C++ OpenMP thread management operate as **independent thread pools**:

- **Julia Task Threads (`julia -t N` / `Threads.nthreads()`)**: Governs concurrency inside Julia scripts (e.g. `Threads.@threads`, `Threads.@spawn`).
- **OpenMP C++ Worker Threads (`OMP_NUM_THREADS=M`)**: Governs parallelism inside C++ kernels (matrix-free CSR apply $y = H \cdot x$, parallel dot products, norms, and Krylov solver iterations).

> **Important**: Starting Julia with `julia -t 1` only constrains Julia's internal thread count. If `OMP_NUM_THREADS` is unset, OpenMP will automatically spawn threads on **all available CPU cores** during matrix-free operations and Krylov iterations.

### 6.2 Setting OpenMP Worker Count

#### Terminal Launch
```bash
# Force single-core execution (1 Julia thread, 1 OpenMP thread)
OMP_NUM_THREADS=1 julia -t 1

# Multi-core C++ execution with single-threaded Julia driver
OMP_NUM_THREADS=8 julia -t 1
```

#### Inside Julia Scripts
```julia
# Must be set before QuantumKrylov initializes the C++ backend
ENV["OMP_NUM_THREADS"] = "4"
using QuantumKrylov
```

### 6.3 Concurrency Patterns & Best Practices

| Use Case | Recommended Settings | Rationale |
| :--- | :--- | :--- |
| **Large Matrix Solver** (Single calculation) | `OMP_NUM_THREADS=all_cores`<br>`julia -t 1` | Maximizes memory bandwidth and vector parallelism in Kokkos OpenMP kernels. |
| **Parameter Sweeps / Parallel Search** | `OMP_NUM_THREADS=1`<br>`julia -t N` | Prevents CPU thread oversubscription and context-switching overhead from nested parallel loops. |

### 6.4 OpenMP Environment Optimizations
For optimal cache locality and NUMA performance on modern multi-core processors:
```bash
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
```

### 6.5 Hardware & Device Query API

`QuantumKrylov.jl` provides runtime utilities to query hardware acceleration and manage execution targets:

```julia
using QuantumKrylov

# 1. Check if backend was compiled with GPU acceleration (CUDA, HIP, SYCL)
is_gpu = is_gpu_build() # Returns Bool (false on CPU-only build)

# 2. Query compiled GPU backend name
backend = find_gpu()    # "cuda", "hip", "sycl", or nothing

# 3. Query physical GPU device count detected on host
num_gpus = gpu_count()  # Returns Int

# 4. Explicitly initialize Kokkos execution spaces for a targeted device
initialize_device!("cpu")
```

| Function | Arguments | Return Type | Description |
| :--- | :--- | :--- | :--- |
| `is_gpu_build()` | None | `Bool` | Returns `true` if compiled with GPU acceleration, `false` otherwise. |
| `find_gpu()` | None | `Union{String, Nothing}` | Returns the active GPU backend name, or `nothing` for CPU builds. |
| `gpu_count()` | None | `Int` | Returns the number of physical GPUs detected on the system. |
| `initialize_device!(device)` | `device::AbstractString="cpu"` | `Nothing` | Explicitly initializes Kokkos execution spaces for `device`. |

### 6.6 Complete GPU Acceleration Workflow

The following example demonstrates an end-to-end workflow querying hardware capabilities, targeting an NVIDIA GPU device, and running Lanczos, Davidson, and Dynamical response solvers:

```julia
using QuantumKrylov

# 1. Hardware detection & device configuration
if is_gpu_build()
    backend = find_gpu()
    n_gpus  = gpu_count()
    println("Accelerated backend active: $backend with $n_gpus physical GPU(s)")

    # Select target device (e.g. "cuda:0" for the first GPU)
    target_device = "cuda:0"
    initialize_device!(target_device)
else
    println("Running on CPU with OpenMP backend")
    target_device = "cpu"
end

# 2. Setup symmetry sector and basis
# 12-site spin-1/2 chain with Sz = 0 symmetry (Hilbert space dimension = 924)
sec = Sector()
set_sz!(sec, 0)
basis = SpinHalfBasis(12, sec)

# 3. Define Heisenberg Hamiltonian terms
op = OpSum()
for i in 0:10
    # Note: `global` is needed when running as a top-level script
    global op += 1.0 * Sz(i) * Sz(i + 1) + 0.5 * (Sp(i) * Sm(i + 1) + Sm(i) * Sp(i + 1))
end

# 4. Construct MatrixFreeHamiltonian on target device
H = MatrixFreeHamiltonian(basis, op; device=target_device)
println("Constructed Hamiltonian on: ", target_device, " (dimension = ", dimension(H), ")")

# 5. Execute Solvers (All computations run directly on target device)
# Ground state calculation:
res_gs = lanczos_ground_state(H, return_state=true, tol=1e-12)
println("Ground state energy: ", res_gs.energy)

# Low-lying excited states using Davidson:
res_dav = davidson_lowest(H, n_eig=3, tol=1e-8)
println("Lowest 3 eigenvalues: ", res_dav.eigenvalues)

# Dynamical spectral function calculation:
v0 = H * res_gs.state
cfr = continued_fraction_coeffs(H, v0, n_iter=50)
I_omega = evaluate_spectral_function(cfr, 0.5, res_gs.energy, 0.05)
println("Spectral function I(omega=0.5): ", I_omega)
```



