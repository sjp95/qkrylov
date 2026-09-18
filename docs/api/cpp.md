# Pure C++20 Developer & API Reference Guide for `qkrylov`

`qkrylov` is a high-performance C++20 matrix-free Krylov subspace diagonalization and quantum dynamics library designed for quantum many-body lattice models. The library is accelerated across heterogeneous architectures (multicore CPUs and GPUs) via **Kokkos**, supports dual precision ([`qkrylov::fp64`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/core/types.hpp#L32-L50) and [`qkrylov::fp32`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/core/types.hpp#L32-L50)), provides an expressive embedded domain-specific language (DSL) for operator algebras, and features compile-time policy-dispatched eigensolvers.

---

## Table of Contents

1. [Architectural Overview & Design Principles](#1-architectural-overview--design-principles)
2. [Source Tree & Header Organization](#2-source-tree--header-organization)
3. [Dual-Precision C++ Core & Type Layout](#3-dual-precision-c-core--type-layout)
4. [Kokkos Hardware Acceleration & Device Management](#4-kokkos-hardware-acceleration--device-management)
5. [Quantum Symmetries & Sector Constraints](#5-quantum-symmetries--sector-constraints)
6. [Hilbert Space Basis Architecture (Fock Representations)](#6-hilbert-space-basis-architecture-fock-representations)
7. [Physical Site Models & Local Action Evaluation](#7-physical-site-models--local-action-evaluation)
8. [Symbolic Operators, Terms & Operator Sum Algebra (`OpSum`)](#8-symbolic-operators-terms--operator-sum-algebra-opsum)
9. [Matrix-Free Hamiltonian Engine (`MatrixFreeHamiltonian`)](#9-matrix-free-hamiltonian-engine-matrixfreehamiltonian)
10. [Kokkos Parallel BLAS-1 Linear Algebra & Tridiagonal QR Solver](#10-kokkos-parallel-blas-1-linear-algebra--tridiagonal-qr-solver)
11. [Eigensolvers & Quantum Dynamics](#11-eigensolvers--quantum-dynamics)
    - [11.1 Compile-Time Policy-Dispatched Lanczos Eigensolver](#111-compile-time-policy-dispatched-lanczos-eigensolver)
    - [11.2 Block Davidson Solver & Diagonal Preconditioning](#112-block-davidson-solver--diagonal-preconditioning)
    - [11.3 Dynamical Continued Fraction Spectroscopy](#113-dynamical-continued-fraction-spectroscopy)
    - [11.4 Finite-Temperature Lanczos Method (FTLM: Dual Workflows)](#114-finite-temperature-lanczos-method-ftlm-dual-workflows)
    - [11.5 Pure-State Real-Time Dynamics (time_evolve)](#115-pure-state-real-time-dynamics-time_evolve)
    - [11.6 Finite-Temperature Dynamical Correlators (ftlm_dynamics)](#116-finite-temperature-dynamical-correlators-ftlm_dynamics)
    - [11.7 Correction Vector Method (Resonance Spectroscopy)](#117-correction-vector-method-resonance-spectroscopy)
12. [Comprehensive End-to-End C++20 Usage Examples](#12-comprehensive-end-to-end-c20-usage-examples)
13. [CMake Build System, Compiler Configuration & Best Practices](#13-cmake-build-system-compiler-configuration--best-practices)

---

## 1. Architectural Overview & Design Principles

### 1.1 Matrix-Free Quantum Many-Body Computation
In quantum many-body lattice models of $N$ sites, the Hilbert space dimension grows exponentially:
$$D = d^N$$
where $d$ is the local site dimension ($d=2$ for spin-$1/2$ or spinless fermions, $d=3$ for spin-$1$ or $t\text{-}J$, $d=4$ for Hubbard electrons). Storing an explicit $D \times D$ matrix requires prohibitive $\mathcal{O}(D^2)$ storage. Even standard sparse CSR/CSC formats demand $\approx 10\text{--}50 \times D$ non-zero entries, exhausting device memory at moderate system sizes ($N \approx 20\text{--}30$).

`qkrylov` computes matrix-vector multiplication $y = \hat{H}x$ on-the-fly:
1. **One-Time Pre-Compilation**: During construction of [`MatrixFreeHamiltonian`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/hamiltonian/matrix_free_hamiltonian.hpp#L55-L128), operator terms are mapped into a compacted Compressed Sparse Row (CSR) structure on device memory (`Kokkos::View`).
2. **Gather-Based Parallel SpMV**: Each thread in parallel computes a row dot product:
   $$y[\alpha] = \sum_{j \in \text{row}(\alpha)} V_j \cdot x[C_j]$$
   Because every thread writes exclusively to its assigned row $\alpha$, execution is **completely atomic-free**.
3. **Zero Allocation in Iterative Loops**: Persistent device scratch views eliminate allocation overhead during solver iterations.

### 1.2 Modern C++20 Paradigm
The codebase utilizes modern C++20 language features:
* **Concepts & Constraints (`requires`)**: Constrains class template argument deduction (CTAD) and execution space dispatching.
* **Structured Bindings**: Results from solvers unpack cleanly (`auto [E0, psi0] = lanczos_ground_state(H);`).
* **Compile-Time Execution Policies**: Governs memory footprints and algorithmic paths via tag dispatch (`policy::OnePass`, `policy::OnePass_DKGS`, `policy::OnePass_full`, `policy::TwoPass`).
* **Strong Typing**: Eliminates ambiguous integer/boolean parameter passing through dedicated tag structs (`sector::Sz`, `sector::Particles`, `device::cpu`, `device::gpu`).

---

## 2. Source Tree & Header Organization

All public C++ headers are in [`include/qkrylov/`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov), and compiled C++ implementations are in [`src/`](file:///home/pritipriya/Documents/GitHub/qkrylov/src):

```text
include/qkrylov/
├── core/
│   ├── types.hpp              # StateID, Index, popcount, Real, Complex, HostVector
│   ├── kokkos_types.hpp       # KComplex, Vector, VectorView, ExecutionSpace, MemorySpace
│   ├── device.hpp             # Device struct, device:: tags, Kokkos init helpers
│   └── traits.hpp             # Execution space traits and tag dispatchers
├── symmetry/
│   └── sector.hpp             # Sector, Sz, Particles, Hubbard, Bosons, Unconstrained
├── basis/
│   ├── basis.hpp              # Abstract Basis base class
│   ├── spinhalf_basis.hpp     # SpinHalfBasis (S=1/2, O(1) zero-RAM implicit indexing)
│   ├── spin_s_basis.hpp       # SpinSBasis (arbitrary spin magnitude S)
│   ├── fermion_basis.hpp      # FermionBasis (spinless fermions, particle conservation)
│   ├── hubbard_basis.hpp      # HubbardBasis (spinful electrons, (N_up, N_dn) conservation)
│   └── tj_basis.hpp           # TJBasis (t-J model with no double occupancy)
├── sites/
│   ├── site.hpp               # Abstract Site base class
│   ├── spinhalf_site.hpp      # SpinHalfSite (Sz, Sp, Sm, Sx, Sy)
│   ├── spin_s_site.hpp        # SpinSSite (arbitrary S local operators)
│   ├── fermion_site.hpp       # FermionSite (N, Id, C, Cdag with Jordan-Wigner phase)
│   ├── hubbard_site.hpp       # HubbardSite (Nup, Ndn, Nupdn, CUp, CdagUp, CDn, CdagDn)
│   └── tj_site.hpp            # TJSite (Gutzwiller-projected electron operators)
├── operators/
│   ├── local_action.hpp       # LocalAction struct (valid, new_state, matrix_element)
│   ├── operator_term.hpp      # OperatorFactor, OperatorTerm, LocalOp, OpSumExpr
│   ├── local_op.hpp           # Free algebraic helpers: Sz(i), Sp(i), Sm(i), etc.
│   ├── symbolic.hpp           # Unified operator header
│   └── opsum.hpp              # OpSum container for symbolic Hamiltonians
├── hamiltonian/
│   └── matrix_free_hamiltonian.hpp # MatrixFreeHamiltonian<ExecSpace>, Hamiltonian alias, CTAD
├── linalg/
│   ├── vector_ops.hpp         # Parallel Kokkos BLAS-1: dot, norm, axpy, scal, normalize
│   └── tridiag_qr.hpp         # Implicit QR tridiagonal eigensolver with Wilkinson shifts
└── solvers/
    ├── policy.hpp             # Lanczos policies: OnePass, OnePass_DKGS, OnePass_full, TwoPass
    ├── lanczos.hpp            # Unified policy-based Lanczos solver and wrappers
    ├── davidson.hpp           # Block Davidson solver with diagonal preconditioning
    ├── dynamics.hpp           # Continued fraction dynamical Green's function
    ├── ftlm.hpp               # Finite-Temperature Lanczos Method (2-stage decoupled)
    └── correction_vector.hpp  # Target frequency correction vector solver
```

---

## 3. Dual-Precision C++ Core & Type Layout

Defined in [`include/qkrylov/core/types.hpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/core/types.hpp#L13-L50):

### 3.1 Precision-Agnostic Types
* [`qkrylov::StateID`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/core/types.hpp#L15): `uint64_t` bitstring representing a quantum Fock state configuration.
* [`qkrylov::Index`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/core/types.hpp#L16): `std::size_t` representing Hilbert space basis dimensions and indices.
* [`qkrylov::popcount(uint64_t x)`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/core/types.hpp#L18-L24): Hardware-accelerated bit population counter (`__builtin_popcountll` on GCC/Clang, `__popcnt64` on MSVC).

### 3.2 Dual Precision Namespaces
Depending on whether `QKRYLOV_DOUBLE_PRECISION` or `QKRYLOV_SINGLE_PRECISION` is defined, precision-dependent types are placed in:
- [`qkrylov::fp64`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/core/types.hpp#L32-L50) (`Real = double`, `Complex = std::complex<double>`)
- [`qkrylov::fp32`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/core/types.hpp#L32-L50) (`Real = float`, `Complex = std::complex<float>`)

```cpp
namespace qkrylov::fp64 {
    using Real       = double;
    using Complex    = std::complex<double>;
    using HostVector = std::vector<Complex>;
}

namespace qkrylov::fp32 {
    using Real       = float;
    using Complex    = std::complex<float>;
    using HostVector = std::vector<Complex>;
}
```

### 3.3 Dynamic Machine-Epsilon Scaling
Algorithms in `qkrylov` avoid hardcoding double-precision thresholds like `1e-14` or `1e-15`. Hardcoded double thresholds stall single-precision execution because $\epsilon_{\text{mach}}(\text{float}) \approx 1.19 \times 10^{-7}$.

The library consistently scales deflation and breakdown criteria via:
```cpp
const Real eps = std::numeric_limits<Real>::epsilon() * Real(4.0);
if (std::abs(e[i]) <= eps * (std::abs(d[i]) + std::abs(d[i+1]))) {
    e[i] = Real(0.0);
}
```

---

## 4. Kokkos Hardware Acceleration & Device Management

Defined in [`include/qkrylov/core/kokkos_types.hpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/core/kokkos_types.hpp#L7-L30), [`include/qkrylov/core/device.hpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/core/device.hpp#L18-L103), and [`include/qkrylov/core/traits.hpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/core/traits.hpp#L7-L85).

### 4.1 Device-Side Complex & View Types
* [`KComplex`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/core/kokkos_types.hpp#L12): `Kokkos::complex<Real>`, binary layout-compatible with `std::complex<Real>`.
* [`VectorView<ExecSpace>`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/core/kokkos_types.hpp#L18): `Kokkos::View<KComplex*, typename ExecSpace::memory_space>`.
* [`Vector`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/core/kokkos_types.hpp#L15): `Kokkos::View<KComplex*>` on the default execution space.

### 4.2 Hardware Device Tags
Tag structs in `namespace qkrylov::device`:
* [`device::cpu`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/core/device.hpp#L20): Targets CPU execution (OpenMP or Serial).
* [`device::gpu`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/core/device.hpp#L21): Targets GPU execution (CUDA, HIP, or SYCL; falls back to CPU if no GPU backend is compiled).
* Specific tags: [`device::openmp`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/core/device.hpp#L23), [`device::serial`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/core/device.hpp#L24), [`device::cuda`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/core/device.hpp#L25), [`device::hip`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/core/device.hpp#L26), [`device::sycl`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/core/device.hpp#L27).

### 4.3 Device Traits Mapping
[`traits::device_execution_space_t<Tag>`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/core/traits.hpp#L78) maps tags to concrete Kokkos execution spaces at compile-time:
```cpp
using Space = qkrylov::traits::device_execution_space_t<qkrylov::device::gpu>;
// Resolves to Kokkos::Cuda, Kokkos::HIP, or Kokkos::Experimental::SYCL
```

### 4.4 Initialization Helpers
* [`qkrylov::detail::ensure_kokkos_initialized()`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/core/device.hpp#L110-L119): Initializes Kokkos if not already initialized, registering `Kokkos::finalize()` via `std::atexit`.
* [`qkrylov::detail::initialize_kokkos(dev)`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/core/device.hpp#L124-L135): Initializes Kokkos targeting a specific device ID (`dev.id`).

---

## 5. Quantum Symmetries & Sector Constraints

Conserving Abelian symmetries restricts the active Hilbert space to a block sector, reducing dimension from $d^N$ to a fraction:

Defined in [`include/qkrylov/symmetry/sector.hpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/symmetry/sector.hpp#L8-L84):

* [`sector::Unconstrained`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/symmetry/sector.hpp#L11): Full Hilbert space without symmetry constraints.
* [`sector::Sz`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/symmetry/sector.hpp#L14-L19): Total $S^z$ conservation. Internally tracks $2 \times S^z$ (`sz2`) to represent half-integers with exact integer arithmetic:
  ```cpp
  sector::Sz sz0(0);       // Sz = 0 (sz2 = 0)
  sector::Sz sz_half(0.5); // Sz = 0.5 (sz2 = 1)
  ```
* [`sector::Particles`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/symmetry/sector.hpp#L22-L26) (alias `sector::ParticleNumber`): Conserves total fermion number $N = \sum_i n_i$.
* [`sector::Hubbard`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/symmetry/sector.hpp#L30-L35): Conserves spin-up and spin-down particle numbers $(N_\uparrow, N_\downarrow)$:
  ```cpp
  sector::Hubbard hub(2, 2); // 2 up, 2 down
  ```
* [`sector::Bosons`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/symmetry/sector.hpp#L38-L42): Conserves total boson number $N_b$.
* [`Sector`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/symmetry/sector.hpp#L46-L84): Unified sector structure supporting multi-quantum-number constraints.

Convenience namespace alias: `basis::sector` is aliased directly to `qkrylov::sector`.

---

## 6. Hilbert Space Basis Architecture (Fock Representations)

The abstract class [`Basis`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/basis/basis.hpp#L7-L20) defines the indexing interface:
```cpp
class Basis {
public:
    virtual ~Basis() = default;
    virtual Index size() const = 0;
    virtual StateID state(Index i) const = 0;
    virtual Index index(StateID s) const = 0;
    virtual bool contains(StateID s) const = 0;
};
```

### 6.1 [`SpinHalfBasis`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/basis/spinhalf_basis.hpp#L13-L70)
* **Model**: Spin-$1/2$ systems ($d=2$). Site $i$ spin-up is bit $i=1$; spin-down is bit $i=0$.
* **Size**: $1 \le N \le 63$ sites.
* **$\mathcal{O}(1)$ Implicit Indexing Optimization**:
  When `sector.use_sz == false`, the internal `states_` vector remains completely unallocated (**zero RAM**). Basis indexing is identity mapping:
  $$\text{state}(i) = i, \quad \text{index}(s) = s$$
* **$S^z$ Constrained Sectors**:
  States matching $2S^z = \text{popcount}(s) - (N - \text{popcount}(s))$ are generated and sorted. `index(s)` executes in $\mathcal{O}(\log D)$ via `std::lower_bound`.

### 6.2 [`SpinSBasis`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/basis/spin_s_basis.hpp#L14-L68)
* **Model**: Arbitrary spin magnitude $S \in \{1/2, 1, 3/2, 2, \dots\}$. Local dimension $d = 2S + 1$.
* **Radix-$d$ State Encoding**:
  Fock states are represented in base-$d$:
  $$s = \sum_{i=0}^{N-1} m_i \cdot d^i, \quad m_i \in \{0, \dots, d-1\}$$
  Local magnetization is $S_i^z = m_i - S$.

### 6.3 [`FermionBasis`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/basis/fermion_basis.hpp#L13-L65)
* **Model**: Spinless fermions ($d=2$). Bit $i=1$ indicates an occupied orbital.
* **Symmetry**: Conserves total particle number $N_e = \text{popcount}(s)$.

### 6.4 [`HubbardBasis`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/basis/hubbard_basis.hpp#L13-L65)
* **Model**: Fermi-Hubbard electrons ($d=4$: empty, $\uparrow$, $\downarrow$, $\uparrow\downarrow$).
* **Interleaved Bit Representation**:
  - Orbital $i$ spin-up: bit $2i$
  - Orbital $i$ spin-down: bit $2i + 1$
* **Capacity**: $1 \le N \le 31$ sites ($2N \le 62$ bits).
* **Symmetry**: Conserves $(N_\uparrow, N_\downarrow)$ independently via masks `up_mask` and `dn_mask`.

### 6.5 [`TJBasis`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/basis/tj_basis.hpp#L13-L63)
* **Model**: Strongly correlated $t\text{-}J$ systems ($d=3$: empty, $\uparrow$, $\downarrow$).
* **Double-Occupancy Projection**:
  States containing both $\uparrow$ and $\downarrow$ on any orbital are excluded during basis generation:
  ```cpp
  if (((s >> (2 * i)) & 1ULL) && ((s >> (2 * i + 1)) & 1ULL)) {
      // Forbidden: double occupancy
  }
  ```

---

## 7. Physical Site Models & Local Action Evaluation

The abstract base class [`Site`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/sites/site.hpp#L13-L24) evaluates local operator action:
```cpp
virtual LocalAction apply(const std::string& op, int site, StateID state) const = 0;
```
[`LocalAction`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/operators/local_action.hpp#L9-L16) returns:
* `bool valid`: `true` if state is not annihilated.
* `StateID new_state`: updated Fock state bitstring.
* `Complex matrix_element`: transition matrix element.

### Concrete Site Classes & Supported Operators

| Site Class | Model | Supported Operators (`op`) | Algebra & Phase Tracking |
| :--- | :--- | :--- | :--- |
| [`SpinHalfSite`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/sites/spinhalf_site.hpp#L11-L28) | Spin-$1/2$ | `"Sz"`, `"Sp"`, `"Sm"`, `"Sx"`, `"Sy"` | $S^z|\uparrow\rangle = +\frac{1}{2}|\uparrow\rangle$, $S^+|\downarrow\rangle = |\uparrow\rangle$, $S^-|\uparrow\rangle = |\downarrow\rangle$. |
| [`SpinSSite`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/sites/spin_s_site.hpp#L11-L30) | Arbitrary $S$ | `"Sz"`, `"Sp"`, `"Sm"`, `"Sx"`, `"Sy"` | $\langle m \pm 1 | S^\pm | m \rangle = \sqrt{S(S+1) - m(m \pm 1)}$. |
| [`FermionSite`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/sites/fermion_site.hpp#L11-L34) | Spinless Fermions | `"N"`, `"Id"`, `"C"`, `"Cdag"` | **Jordan-Wigner phase**: $(-1)^{\sum_{j < i} n_j}$ computed using `popcount(state & ((1ULL << site) - 1))`. |
| [`HubbardSite`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/sites/hubbard_site.hpp#L11-L35) | Fermi-Hubbard | `"Nup"`, `"Ndn"`, `"Nupdn"`, `"CUp"`, `"CdagUp"`, `"CDn"`, `"CdagDn"` | Separate Jordan-Wigner phase strings for $\uparrow$ and $\downarrow$. |
| [`TJSite`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/sites/tj_site.hpp#L11-L35) | $t\text{-}J$ Model | `"Nup"`, `"Ndn"`, `"CUp"`, `"CdagUp"`, `"CDn"`, `"CdagDn"` | `CdagUp` and `CdagDn` annihilate the state if the site is already occupied by *either* spin. |

Automatic site deduction from a basis:
```cpp
std::shared_ptr<Site> site = infer_site_from_basis(basis);
```

---

## 8. Symbolic Operators, Terms & Operator Sum Algebra (`OpSum`)

### 8.1 Operator Data Structures
Defined in [`include/qkrylov/operators/operator_term.hpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/operators/operator_term.hpp#L13-L116) and [`include/qkrylov/operators/opsum.hpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/operators/opsum.hpp#L13-L48):

* [`OperatorFactor`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/operators/operator_term.hpp#L13-L20): `(std::string op, int site)`.
* [`OperatorTerm`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/operators/operator_term.hpp#L22-L32): `(Complex coeff, std::vector<OperatorFactor> factors)`.
* [`LocalOp`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/operators/operator_term.hpp#L37-L43): Named site operator representation for expression templates.
* [`OpSum`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/operators/opsum.hpp#L13-L48): Vector of operator terms defining the Hamiltonian.

### 8.2 Algebraic Expression Template Syntax
Helper generator functions in [`include/qkrylov/operators/local_op.hpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/operators/local_op.hpp#L12-L30):
* Spin: `Sz(i)`, `Sp(i)`, `Sm(i)`, `Sx(i)`, `Sy(i)`
* Fermion/Hubbard: `CdagUp(i)`, `CUp(i)`, `CdagDn(i)`, `CDn(i)`, `Nup(i)`, `Ndn(i)`, `Nupdn(i)`
* Boson: `Bdag(i)`, `B(i)`, `N(i)`

Operators overload arithmetic operations:
```cpp
OpSum ops;

// Heisenberg interaction
ops += 1.0 * Sz(0) * Sz(1) + 0.5 * Sp(0) * Sm(1) + 0.5 * Sm(0) * Sp(1);

// Transverse field
ops += -0.5 * Sx(0) - 0.5 * Sx(1);
```

---

## 9. Matrix-Free Hamiltonian Engine (`MatrixFreeHamiltonian`)

Defined in [`include/qkrylov/hamiltonian/matrix_free_hamiltonian.hpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/hamiltonian/matrix_free_hamiltonian.hpp#L55-L168) and implemented in [`src/hamiltonian/matrix_free_hamiltonian.cpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/src/hamiltonian/matrix_free_hamiltonian.cpp#L13-L263).

### 9.1 Construction & Pre-Compiled CSR Structure
Construction executes in two phases:
1. **Phase 1 (Host Build)**: Iterates over all basis states $\alpha \in [0, D)$, evaluates the operator action via `site->apply`, resolves target indices $\beta = \text{basis->index}(s')$, sorts and sums duplicate non-zeros per row, and extracts diagonal entries $D_{\alpha\alpha} = \langle \alpha | \hat{H} | \alpha \rangle$.
2. **Phase 2 (Device Transfer)**: Deep-copies `row_offsets_`, `col_indices_`, `values_`, and `diagonal_` into device `Kokkos::View` allocations.

### 9.2 Gather-Based Atomic-Free SpMV
Matrix-vector multiplication computes $y = \hat{H}x$ using parallel gather:
```cpp
Kokkos::parallel_for("qkrylov::H_apply",
    Kokkos::RangePolicy<ExecSpace, Index>(0, dim),
    KOKKOS_LAMBDA(const Index alpha) {
        KComplex sum(0.0, 0.0);
        const Index row_begin = rows(alpha);
        const Index row_end   = rows(alpha + 1);
        for (Index j = row_begin; j < row_end; ++j) {
            sum += vals(j) * x(cols(j));
        }
        y(alpha) = sum;
    }
);
```

### 9.3 Class Template Interface & CTAD Guides
```cpp
template <typename ExecSpace = Kokkos::DefaultExecutionSpace>
class MatrixFreeHamiltonian {
public:
    // Device-resident zero-copy multiplication
    void apply(const VectorView<ExecSpace>& x, VectorView<ExecSpace>& y) const;

    // Host-pointer multiplication (uses cached scratch device views)
    void apply(const Complex* x, Complex* y) const;

    // Diagonal access
    const VectorView<ExecSpace>& diagonal() const;
    HostVector diagonal_host() const;

    Index dimension() const;
    const Device& device() const;
};

// Convenient template alias
template <typename ExecSpace = Kokkos::DefaultExecutionSpace>
using Hamiltonian = MatrixFreeHamiltonian<ExecSpace>;
```

CTAD deduction guides automatically deduce the Kokkos execution space and infer the local physical site:
```cpp
auto basis = basis::SpinHalf(4, basis::sector::Sz{0});
OpSum ops;

Hamiltonian H(basis, ops);                // Deduces CPU execution space, infers SpinHalfSite
Hamiltonian H_gpu(basis, ops, device::gpu{}); // Deduces GPU execution space
```

---

## 10. Kokkos Parallel BLAS-1 Linear Algebra & Tridiagonal QR Solver

### 10.1 Parallel Vector Operations ([`vector_ops.hpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/linalg/vector_ops.hpp#L13-L137))
All kernels operate on device `VectorView<ExecSpace>` views:
* [`dot(x, y)`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/linalg/vector_ops.hpp#L14-L32): Complex inner product $\langle x | y \rangle = \sum_i x_i^* y_i$ using parallel reduction.
* [`norm(x)`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/linalg/vector_ops.hpp#L35-L43): Euclidean 2-norm $\sqrt{\text{Re}\langle x | x \rangle}$.
* [`scal(a, x)`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/linalg/vector_ops.hpp#L46-L69): Vector scaling $x \leftarrow a \cdot x$ (overloaded for `KComplex` and `Real`).
* [`axpy(a, x, y)`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/linalg/vector_ops.hpp#L72-L86): Fused multiply-add $y \leftarrow a \cdot x + y$.
* [`normalize(x)`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/linalg/vector_ops.hpp#L89-L97): In-place unit normalization $x \leftarrow x / \|x\|$.
* [`zero_fill(x)`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/linalg/vector_ops.hpp#L100-L104): Fills device vector with zero.
* [`deep_copy(dst, src)`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/linalg/vector_ops.hpp#L107-L111): Deep copy between device views.
* [`copy_host_to_device(host, dev)`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/linalg/vector_ops.hpp#L114-L124) / [`copy_device_to_host(dev, host)`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/linalg/vector_ops.hpp#L127-L137): Host $\leftrightarrow$ device transfers.

### 10.2 Tridiagonal QR Eigensolver ([`tridiag_qr.hpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/linalg/tridiag_qr.hpp#L14-L130))
Diagonalizes the $m \times m$ symmetric tridiagonal matrix $T_m$ resulting from Lanczos recurrence:
* Implements **implicit QR iterations with Wilkinson shifts** and Givens plane rotations.
* Guarantees monotonic convergence and sorted eigenvalues.
* [`linalg::tridiag_eigensystem_full(alphas, betas, m)`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/linalg/tridiag_qr.hpp#L27-L119): Returns all eigenvalues and eigenvectors.
* [`linalg::tridiag_ground_state_full(alphas, betas, m)`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/linalg/tridiag_qr.hpp#L122-L130): Ground state fast path.

---

## 11. Eigensolvers & Quantum Dynamics

### 11.1 Compile-Time Policy-Dispatched Lanczos Eigensolver

The central solver [`solvers::lanczos`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/solvers/lanczos.hpp#L95-L98) takes a compile-time policy tag from `namespace qkrylov::solvers::policy`:

```cpp
template <typename Policy = policy::Default, typename ExecSpace>
LanczosResult lanczos(const MatrixFreeHamiltonian<ExecSpace>& H, const LanczosConfig& config = {});
```

#### Policy Characteristics

| Policy | Scope | Memory Footprint | DGKS Reorthogonalization | Eigenvector Computation |
| :--- | :--- | :--- | :--- | :--- |
| [`policy::OnePass`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/solvers/policy.hpp#L10) | Ground state energy only ($n_{\text{eig}}=1$) | Strictly $3N$ scalars ($\mathcal{O}(N)$) | Disabled (ghost states cannot drop below $E_0$) | None (`res.eigenvector.empty() == true`) |
| [`policy::OnePass_DKGS`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/solvers/policy.hpp#L11) | Arbitrary low-lying states ($n_{\text{eig}} \ge 1$) | $m \cdot N$ scalars ($\mathcal{O}(m \cdot N)$) | **2-pass DGKS enabled** (machine epsilon precision) | Exact Ritz vectors for all $n_{\text{eig}}$ states |
| [`policy::OnePass_full`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/solvers/policy.hpp#L12) | Ground state + vector ($n_{\text{eig}}=1$) | $m \cdot N$ scalars ($\mathcal{O}(m \cdot N)$) | Disabled | Fast single-pass accumulation $|\psi_0\rangle = \sum_j y_j^{(0)} v_j$ |
| [`policy::TwoPass`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/solvers/policy.hpp#L13) | Ground state + vector ($n_{\text{eig}}=1$) | Strictly $4N$ scalars ($\mathcal{O}(N)$) | Disabled | **Pass 2 seed/vector replay** with SpMV elision |

#### Solver Configuration & Results
Defined in [`include/qkrylov/solvers/lanczos.hpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/solvers/lanczos.hpp#L16-L64):

* [`LanczosConfig`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/solvers/lanczos.hpp#L16-L27):
  ```cpp
  struct LanczosConfig {
      int n_eig = 1;                               // Target low-lying states (OnePass_DKGS)
      int maxiter = 200;                           // Max Krylov iterations
      int min_iterations = 1;                      // Steps before checking convergence
      int check_interval = 1;                      // Convergence check stride
      Real tol = Real(1.0e-12);                    // Ritz residual tolerance
      Real breakdown_tol = Real(0.0);              // Invariant subspace tolerance (0 => 4 * eps)
      std::optional<uint64_t> seed = std::nullopt; // Deterministic PRNG seed (default: 123456789ULL)
      HostVector initial_vector = {};              // Warm-starting trial state
      bool compute_eigenvectors = true;            // Toggle eigenvector accumulation
  };
  ```
* [`LanczosResult`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/solvers/lanczos.hpp#L29-L64):
  Contains `energy`, `eigenvalues`, `eigenvector`, `eigenvectors`, `iterations`, `converged`, `alphas`, `betas`.
  Supports **structured bindings**:
  ```cpp
  auto [E0, psi0] = solvers::lanczos<solvers::policy::TwoPass>(H, cfg);
  ```

#### Idiomatic Convenience Wrappers
```cpp
auto res_e0    = lanczos_energy(H);                  // OnePass (O(N) memory, scalar E0)
auto res_gs    = lanczos_ground_state(H);            // OnePass_DKGS
auto res_tp    = lanczos_two_pass(H);                // TwoPass (O(N) memory + eigenvector, n_eig = 1)
auto res_multi = lanczos_lowest(H, /* n_eig = */ 4); // OnePass_DKGS (4 lowest states)
```

> [!NOTE]
> `solvers::policy::TwoPass` is fundamentally tailored for ground-state eigenvector reconstruction ($n_{\text{eig}} = 1$) with minimal memory footprint by replaying the Lanczos iteration from the deterministic initial seed. For computing multiple low-lying states ($n_{\text{eig}} > 1$), use `solvers::policy::OnePass_DKGS` (`lanczos_lowest`) or the Block Davidson solver (`davidson_lowest`).

---

### 11.2 Block Davidson Solver & Diagonal Preconditioning

Defined in [`include/qkrylov/solvers/davidson.hpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/solvers/davidson.hpp#L13-L28) and implemented in [`src/solvers/davidson.cpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/src/solvers/davidson.cpp):

```cpp
template <typename ExecSpace>
DavidsonResult davidson_lowest(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    int n_eig = 1,
    int max_subspace = 20,
    Real tol = 1.0e-8
);
```

#### The Diagonal Preconditioner Pole
Davidson accelerates convergence for diagonally dominant Hamiltonians using the preconditioner:
$$\delta_i[j] = \frac{r_i[j]}{D_{jj} - \theta_i}$$
where $r_i = \hat{H}u_i - \theta_i u_i$ and $D_{jj} = \langle j | \hat{H} | j \rangle$.

> [!WARNING]
> For excited states ($k \ge 3$), the Ritz eigenvalue $\theta_k$ enters the interior of the diagonal spectrum band, resulting in $D_{jj} - \theta_k \to 0$. The denominator vanishes, causing division-by-zero or numerical blowup.
> **Recommendation**: For multiple excited states ($n_{\text{eig}} > 1$), use **`lanczos_lowest` (`policy::OnePass_DKGS`)**, which has no preconditioner poles.

---

### 11.3 Dynamical Continued Fraction Spectroscopy

Defined in [`include/qkrylov/solvers/dynamics.hpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/solvers/dynamics.hpp#L13-L39) and implemented in [`src/solvers/dynamics.cpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/src/solvers/dynamics.cpp).

Computes the dynamical Green's function response:
$$G(z) = \langle \phi_0 | \frac{1}{z - \hat{H} + E_0} | \phi_0 \rangle, \quad z = \omega + i\eta$$

1. [`continued_fraction_coeffs(H, phi0, n_iter)`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/solvers/dynamics.hpp#L22-L26): Computes recurrence coefficients $\alpha_n, \beta_n$ by delegating directly to `solvers::lanczos<policy::OnePass>`.
2. [`evaluate_spectral_function(...)`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/solvers/dynamics.hpp#L30-L38): Evaluates the continued fraction backward recurrence:
   $$C_n = \frac{1}{z - \alpha_n}, \quad C_i = \frac{1}{z - \alpha_i - \beta_i^2 C_{i+1}}$$
   returning the spectral function $S(\omega) = -\frac{1}{\pi} \text{Im} [\|\phi_0\|^2 C_0]$.

---

### 11.4 Finite-Temperature Lanczos Method (FTLM: Dual Workflows)

Defined in [`include/qkrylov/solvers/ftlm.hpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/solvers/ftlm.hpp#L13-L120) and implemented in [`src/solvers/ftlm.cpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/src/solvers/ftlm.cpp).

Computes finite-temperature thermal averages and thermodynamics over random Gaussian trial states $\{|r\rangle\}$ projected strictly onto the unit sphere $\mathbb{S}^{D-1}$:
$$\langle \hat{O} \rangle_\beta = \frac{1}{Z(\beta)} \operatorname{Tr}\left(e^{-\beta \hat{H}} \hat{O}\right) = \frac{1}{\bar{Z}(\beta)} \frac{1}{R} \sum_{r=1}^R \mathbf{c}^{(r)\dagger} \mathcal{O}^{(r)} \mathbf{c}^{(r)}$$
where $\mathbf{c}^\dagger \mathcal{O} \mathbf{c} = \sum_{j,k} c_j^* \mathcal{O}_{jk} c_k \in \mathbb{C}$ is evaluated in its exact sesquilinear form, returning complex expectations for non-Hermitian and complex Hermitian observables ($S^y, S^+, c_k, J$).

#### Mathematical & Numerical Innovations:
1. **Haar Unit-Sphere Trace Scaling**: Gaussian vector lengths $\|r\|^2 \sim \chi^2(2D)$ fluctuations are eliminated by projecting onto $\mathbb{S}^{D-1}$ and scaling averages by the Hilbert space dimension $D/R$:
   $$Z(\beta) = \frac{D}{R} \sum_{r=1}^R \sum_{m=0}^{M-1} (y_{0m}^{(r)})^2 e^{-\beta(\epsilon_m^{(r)} - E_{\min})}$$
2. **Dual-Workflow Architecture**:
   - **Streamed Two-Pass Mode (`ftlm_sweep_streamed` / `FTLMWorkflow::Streamed`)**: Bounded to $\mathcal{O}(M^2 N_{\text{obs}})$ peak RAM (~16 MB). Pass 1 discovers global shift $E_{\min}$ and caches recurrence coefficients; Pass 2 streams projection on-the-fly for the input `beta_grid` and immediately frees operator matrices per sample.
   - **Cached Mode (`ftlm_sample` + `ftlm_evaluate_sweep` / `FTLMWorkflow::Cached`)**: Preserves Krylov samples and projected matrices in `std::vector<FTLMKrylovSample>` for instant zero-SpMV evaluation across arbitrary temperature grids.
3. **Linearized Ratio Error Bars**: Replaces unstable sample quotients $A_r / Z_r$ with the linearized covariance-aware standard error:
   $$\sigma_{\bar{O}} = \frac{1}{\bar{Z}} \sqrt{\frac{1}{R(R - 1)} \sum_{r=1}^R \left| A_r - \bar{O} Z_r \right|^2}$$
4. **Effective Sample Diagnostic ($R_{\text{eff}}$)**: Detects low-$T$ single-sample freeze-out:
   $$R_{\text{eff}}(\beta) = \frac{\left(\sum_{r=1}^R Z_r(\beta)\right)^2}{\sum_{r=1}^R Z_r(\beta)^2}$$
5. **Thermodynamically Consistent Entropy**: Evaluated via thermodynamic integration anchored at $\lim_{\beta \to 0} S = \ln D$:
   $$S(\beta) = \ln D - \int_0^\beta \beta' C_v(\beta') d\beta' \quad (\text{when } \beta_{\min} \le 10^{-4})$$
   and falls back to state function $S(\beta) = \max(0, \beta(\langle E \rangle - F))$ when sweeping an unanchored finite-temperature window ($\beta_{\min} > 10^{-4}$).

```cpp
// Mode 1: Streamed Two-Pass (Production, O(M^2 * N_obs) RAM)
auto sweep = ftlm_sweep(H, beta_grid, {SzSz}, n_random, n_steps, seed, FTLMWorkflow::Streamed);

// Mode 2: Cached Stage 1 (Sampling) + Stage 2 (Instant Re-evaluation)
auto samples = ftlm_sample(H, {SzSz}, n_random, n_steps, seed);
auto sweep_eval = ftlm_evaluate_sweep(samples, beta_grid, H.dimension());
```

---

### 11.5 Pure-State Real-Time Dynamics (`time_evolve`)

Defined in [`include/qkrylov/solvers/dynamics.hpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/solvers/dynamics.hpp#L40-L75) and implemented in [`src/solvers/dynamics.cpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/src/solvers/dynamics.cpp).

Propagates an arbitrary initial pure state $|\psi(0)\rangle$ under unitary real-time Schrödinger evolution:
$$|\psi(t)\rangle = e^{-i \hat{H} t} |\psi(0)\rangle$$

```cpp
struct RealTimeResult {
    std::vector<Real> time_grid;
    std::vector<Complex> survival_probabilities;             // L(t) = <psi(0)|psi(t)>
    std::vector<std::vector<Complex>> observable_expectations; // [obs_idx][time_idx]
};

template <typename ExecSpace>
RealTimeResult time_evolve(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    const HostVector& psi0,
    const std::vector<Real>& time_grid,
    const std::vector<MatrixFreeHamiltonian<ExecSpace>>& observables = {},
    int n_steps = 30
);
```

- **Short-to-Medium Times ($t \lesssim M / \|H\|$):** One Krylov run evaluates an arbitrary dense time grid $t_1, \dots, t_K$ in milliseconds via scalar matrix contractions.
- **Long Times ($t \gg 1$):** Adaptive Krylov time-stepping updates $|\psi(t + \Delta t)\rangle \approx \exp(-i \hat{H} \Delta t) |\psi(t)\rangle$ with exact norm conservation.

---

### 11.6 Finite-Temperature Real-Time Dynamics (`ftlm_dynamics`)

Defined in [`include/qkrylov/solvers/dynamics.hpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/solvers/dynamics.hpp#L77-L120) and implemented in [`src/solvers/dynamics.cpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/src/solvers/dynamics.cpp).

Computes unequal-time finite-temperature dynamical correlation functions:
$$C_{AB}(t; \beta) = \langle \hat{A}(t) \hat{B}(0) \rangle_\beta = \frac{1}{Z(\beta)} \operatorname{Tr}\left( e^{-\beta \hat{H}} e^{i \hat{H} t} \hat{A} e^{-i \hat{H} t} \hat{B} \right)$$

Uses **dual-Krylov propagation**:
1. Prepares thermal state $|\phi_r(\beta/2)\rangle = e^{-\beta \hat{H}/2} |r\rangle$ in $\mathcal{K}(H, |r\rangle)$.
2. Forms the branching state $|\chi_r(\beta/2)\rangle = \hat{B} |\phi_r(\beta/2)\rangle$ directly in the **full $D$-dimensional Hilbert space**.
3. Launches a second independent Krylov propagation for $|\chi_r\rangle$ in $\mathcal{K}(H, |\chi_r\rangle)$. This completely eliminates operator projection truncation errors and makes $C_{AB}(0) = \langle \hat{A} \hat{B} \rangle$ exact to machine precision.
4. Performs linearized ratio averaging over $R$ samples to yield $C_{AB}(t; \beta)$ and its statistical error bars $\sigma_{C}(t)$.

```cpp
struct FTLMDynamicsResult {
    Real beta;
    std::vector<Real> time_grid;
    std::vector<Complex> correlations;    // C_AB(t) = <A(t) B(0)>_beta
    std::vector<Real> correlation_errors; // Linearized ratio error bars
};

template <typename ExecSpace>
FTLMDynamicsResult ftlm_dynamics(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    Real beta,
    const MatrixFreeHamiltonian<ExecSpace>& A,
    const MatrixFreeHamiltonian<ExecSpace>& B,
    const std::vector<Real>& time_grid,
    int n_random = 50,
    int n_steps = 100,
    uint64_t seed = 42
);
```

---

### 11.7 Correction Vector Method (Resonance Spectroscopy)

Defined in [`include/qkrylov/solvers/correction_vector.hpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/solvers/correction_vector.hpp#L10-L27) and implemented in [`src/solvers/correction_vector.cpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/src/solvers/correction_vector.cpp).

Solves the shifted linear system:
$$\left[ (\hat{H} - E_0 - \omega)^2 + \eta^2 \right] |y\rangle = \eta \hat{A} |0\rangle$$
using an iterative **Conjugate Gradient (CG)** solver without squaring the matrix explicitly. Returns the correction vector $|x\rangle$ and spectral response $S(\omega) = \frac{1}{\pi} \text{Re} \langle \hat{A} 0 | x \rangle$.

---

## 12. Comprehensive End-to-End C++20 Usage Examples

### Example 1: 1D Antiferromagnetic Heisenberg Spin-1/2 Chain (TwoPass Lanczos)
Solves for the ground state energy and state vector using the memory-constrained `TwoPass` policy:

```cpp
#include <qkrylov/basis/spinhalf_basis.hpp>
#include <qkrylov/operators/opsum.hpp>
#include <qkrylov/operators/local_op.hpp>
#include <qkrylov/hamiltonian/matrix_free_hamiltonian.hpp>
#include <qkrylov/solvers/lanczos.hpp>
#include <iostream>

using namespace qkrylov;
using namespace qkrylov::fp64;

int main() {
    const int N = 12;

    // Conserve total Sz = 0
    auto basis = std::make_shared<SpinHalfBasis>(N, Sector(sector::Sz{0}));

    // H = \sum_i [ S^z_i S^z_{i+1} + 0.5(S^+_i S^-_{i+1} + S^-_i S^+_{i+1}) ]
    OpSum ops;
    for (int i = 0; i < N; ++i) {
        int j = (i + 1) % N;
        ops += 1.0 * Sz(i) * Sz(j) + 0.5 * Sp(i) * Sm(j) + 0.5 * Sm(i) * Sp(j);
    }

    // MatrixFreeHamiltonian automatically infers SpinHalfSite
    Hamiltonian H(basis, ops);
    std::cout << "Hilbert space dimension: " << H.dimension() << std::endl;

    LanczosConfig cfg;
    cfg.maxiter = 100;
    cfg.tol = 1e-12;

    // TwoPass uses strictly 4 device vectors in RAM
    auto [E0, psi0] = solvers::lanczos<solvers::policy::TwoPass>(H, cfg);

    std::cout << "Ground state energy: " << E0 << std::endl;
    std::cout << "State vector dimension: " << psi0.size() << std::endl;
    return 0;
}
```

### Example 2: Spin-1 Haldane Chain with Multiple Excited States (OnePass_DKGS)
Calculates the ground state and 3 lowest excited states with full DGKS reorthogonalization:

```cpp
#include <qkrylov/basis/spin_s_basis.hpp>
#include <qkrylov/operators/opsum.hpp>
#include <qkrylov/operators/local_op.hpp>
#include <qkrylov/hamiltonian/matrix_free_hamiltonian.hpp>
#include <qkrylov/solvers/lanczos.hpp>
#include <iostream>

using namespace qkrylov;
using namespace qkrylov::fp64;

int main() {
    const int N = 6;
    const double S = 1.0;

    auto basis = std::make_shared<SpinSBasis>(N, S, Sector(sector::Sz{0}));

    OpSum ops;
    for (int i = 0; i < N - 1; ++i) {
        ops += 1.0 * Sz(i) * Sz(i + 1) + 0.5 * Sp(i) * Sm(i + 1) + 0.5 * Sm(i) * Sp(i + 1);
    }

    Hamiltonian H(basis, ops);

    LanczosConfig cfg;
    cfg.n_eig = 4;
    cfg.maxiter = 150;
    cfg.tol = 1e-10;

    auto res = solvers::lanczos<solvers::policy::OnePass_DKGS>(H, cfg);

    std::cout << "Converged in " << res.iterations << " iterations." << std::endl;
    for (size_t k = 0; k < res.eigenvalues.size(); ++k) {
        std::cout << "E[" << k << "] = " << res.eigenvalues[k] << std::endl;
    }
    return 0;
}
```

### Example 3: Fermi-Hubbard Model with Spin-Resolved Particles (OnePass_full)
```cpp
#include <qkrylov/basis/hubbard_basis.hpp>
#include <qkrylov/sites/hubbard_site.hpp>
#include <qkrylov/operators/opsum.hpp>
#include <qkrylov/operators/local_op.hpp>
#include <qkrylov/hamiltonian/matrix_free_hamiltonian.hpp>
#include <qkrylov/solvers/lanczos.hpp>
#include <iostream>

using namespace qkrylov;
using namespace qkrylov::fp64;

int main() {
    const int N = 4;
    const double t = 1.0;
    const double U = 4.0;

    // Half-filling: 2 up, 2 down
    auto basis = std::make_shared<HubbardBasis>(N, Sector(sector::Hubbard{2, 2}));
    auto site = std::make_shared<HubbardSite>();

    OpSum ops;
    // Hopping: -t \sum_{<i,j>, \sigma} (c^\dagger_{i\sigma} c_{j\sigma} + h.c.)
    for (int i = 0; i < N - 1; ++i) {
        ops += -t * CdagUp(i) * CUp(i + 1) + -t * CdagUp(i + 1) * CUp(i);
        ops += -t * CdagDn(i) * CDn(i + 1) + -t * CdagDn(i + 1) * CDn(i);
    }
    // On-site interaction: U \sum_i n_{i\uparrow} n_{i\downarrow}
    for (int i = 0; i < N; ++i) {
        ops += U * Nupdn(i);
    }

    Hamiltonian H(basis, site, ops);
    std::cout << "Hubbard dimension: " << H.dimension() << std::endl;

    auto [E0, psi0] = solvers::lanczos<solvers::policy::OnePass_full>(H);
    std::cout << "Ground state energy: " << E0 << std::endl;
    return 0;
}
```

### Example 4: Strongly Correlated $t\text{-}J$ Chain with Double-Occupancy Exclusion
```cpp
#include <qkrylov/basis/tj_basis.hpp>
#include <qkrylov/sites/tj_site.hpp>
#include <qkrylov/operators/opsum.hpp>
#include <qkrylov/operators/local_op.hpp>
#include <qkrylov/hamiltonian/matrix_free_hamiltonian.hpp>
#include <qkrylov/solvers/lanczos.hpp>
#include <iostream>

using namespace qkrylov;
using namespace qkrylov::fp64;

int main() {
    const int N = 4;
    const double t = 1.0;
    const double J = 0.5;

    // 1 up electron, 1 down electron, 2 holes
    auto basis = std::make_shared<TJBasis>(N, Sector(sector::Hubbard{1, 1}));
    auto site = std::make_shared<TJSite>();

    OpSum ops;
    // Projected hopping (double occupancy forbidden by TJSite)
    for (int i = 0; i < N - 1; ++i) {
        ops += -t * CdagUp(i) * CUp(i + 1) + -t * CdagUp(i + 1) * CUp(i);
        ops += -t * CdagDn(i) * CDn(i + 1) + -t * CdagDn(i + 1) * CDn(i);
    }
    // Antiferromagnetic exchange interaction: J \sum (S_i \cdot S_{i+1} - 0.25 n_i n_{i+1})
    for (int i = 0; i < N - 1; ++i) {
        ops += J * Sz(i) * Sz(i + 1) + 0.5 * J * Sp(i) * Sm(i + 1) + 0.5 * J * Sm(i) * Sp(i + 1);
        ops += -0.25 * J * (Nup(i) + Ndn(i)) * (Nup(i + 1) + Ndn(i + 1));
    }

    Hamiltonian H(basis, site, ops);
    std::cout << "t-J Hilbert space dimension: " << H.dimension() << std::endl;

    auto [E0, psi0] = solvers::lanczos<solvers::policy::OnePass_full>(H);
    std::cout << "t-J ground state energy: " << E0 << std::endl;
    return 0;
}
```

### Example 5: Finite-Temperature Multi-Temperature Sweep & Observables (FTLM)
```cpp
#include <qkrylov/basis/spinhalf_basis.hpp>
#include <qkrylov/operators/opsum.hpp>
#include <qkrylov/operators/local_op.hpp>
#include <qkrylov/hamiltonian/matrix_free_hamiltonian.hpp>
#include <qkrylov/solvers/ftlm.hpp>
#include <iostream>
#include <vector>

using namespace qkrylov;
using namespace qkrylov::fp64;

int main() {
    const int N = 8;
    auto basis = std::make_shared<SpinHalfBasis>(N);

    // Hamiltonian
    OpSum ops_H;
    for (int i = 0; i < N - 1; ++i) {
        ops_H += 1.0 * Sz(i) * Sz(i + 1) + 0.5 * Sp(i) * Sm(i + 1) + 0.5 * Sm(i) * Sp(i + 1);
    }
    Hamiltonian H(basis, ops_H);

    // Observable: nearest-neighbor spin-spin correlation Sz(0) * Sz(1)
    OpSum ops_corr;
    ops_corr += 1.0 * Sz(0) * Sz(1);
    Hamiltonian O_corr(basis, ops_corr);

    // Stage 1: Sample Krylov subspace with R=60 random vectors
    int n_random = 60;
    int n_steps = 30;
    auto samples = ftlm_sample(H, {O_corr}, n_random, n_steps, /* seed = */ 12345ULL);

    // Stage 2: Instantaneous multi-temperature sweep
    std::vector<Real> beta_grid = {0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0};
    auto sweep = ftlm_evaluate_sweep(samples, beta_grid);

    std::cout << "Beta\tEnergy\t\tCv\t\t<Sz0 Sz1>\tStdError" << std::endl;
    for (size_t i = 0; i < beta_grid.size(); ++i) {
        std::cout << sweep.beta_grid[i] << "\t"
                  << sweep.internal_energies[i] << "\t"
                  << sweep.specific_heats[i] << "\t"
                  << sweep.observable_expectations[0][i] << "\t"
                  << sweep.observable_errors[0][i] << std::endl;
    }
    return 0;
}
```

### Example 6: Dynamical Spectral Function via Correction Vector Method
```cpp
#include <qkrylov/basis/spinhalf_basis.hpp>
#include <qkrylov/operators/opsum.hpp>
#include <qkrylov/operators/local_op.hpp>
#include <qkrylov/hamiltonian/matrix_free_hamiltonian.hpp>
#include <qkrylov/solvers/lanczos.hpp>
#include <qkrylov/solvers/correction_vector.hpp>
#include <iostream>

using namespace qkrylov;
using namespace qkrylov::fp64;

int main() {
    const int N = 8;
    auto basis = std::make_shared<SpinHalfBasis>(N, Sector(sector::Sz{0}));

    OpSum ops;
    for (int i = 0; i < N - 1; ++i) {
        ops += 1.0 * Sz(i) * Sz(i + 1) + 0.5 * Sp(i) * Sm(i + 1) + 0.5 * Sm(i) * Sp(i + 1);
    }
    Hamiltonian H(basis, ops);

    // 1. Find ground state
    auto [E0, psi0] = solvers::lanczos<solvers::policy::OnePass_full>(H);

    // 2. Prepare excitation state |phi0> = S^z_0 |psi0>
    OpSum op_sz0;
    op_sz0 += 1.0 * Sz(0);
    Hamiltonian H_sz0(basis, op_sz0);

    HostVector Op_psi0(H.dimension());
    H_sz0.apply(psi0.data(), Op_psi0.data());

    // 3. Evaluate spectral response at frequency omega = 1.0 with broadening eta = 0.1
    Real omega = 1.0;
    Real eta = 0.1;
    auto cv_res = correction_vector_spectral(H, Op_psi0, E0, omega, eta);

    std::cout << "Correction vector converged: " << (cv_res.converged ? "YES" : "NO")
              << " in " << cv_res.iterations << " iterations." << std::endl;
    std::cout << "Spectral intensity S(omega=" << omega << "): " << cv_res.spectral_function << std::endl;
    return 0;
}
```

---

## 13. CMake Build System, Compiler Configuration & Best Practices

### 13.1 Building with CMake
```bash
# Configure build directory with tests enabled
cmake -B build -S . -DQKRYLOV_BUILD_TESTS=ON

# Compile the library and tests
cmake --build build -j$(nproc)

# Run C++ unit test suite via ctest
cd build && ctest --output-on-failure
```

### 13.2 Linking in an External CMake Project
```cmake
cmake_minimum_required(VERSION 3.20)
project(my_quantum_app LANGUAGES CXX)

find_package(qkrylov CONFIG REQUIRED)

add_executable(my_quantum_app main.cpp)
target_link_libraries(my_quantum_app PRIVATE qkrylov::qkrylov)
target_compile_features(my_quantum_app PRIVATE cxx_std_20)
```

### 13.3 Kokkos Backend Options
During CMake configuration, Kokkos selects the target architecture:
* **OpenMP (Multithreaded CPU)**: `-DKokkos_ENABLE_OPENMP=ON` (enabled by default on non-MSVC systems)
* **NVIDIA CUDA**: `-DKokkos_ENABLE_CUDA=ON -DCMAKE_CXX_COMPILER=nvcc_wrapper`
* **AMD HIP**: `-DKokkos_ENABLE_HIP=ON`
* **Intel SYCL**: `-DKokkos_ENABLE_SYCL=ON`

### 13.4 Developer Best Practices Checklist
1. **Dynamic Epsilon Scaling**: Never compare floating-point residuals against static double tolerances like `1e-14`. Use `std::numeric_limits<Real>::epsilon() * 4.0` so algorithms function identically in both `fp32` and `fp64`.
2. **Select Policies by Problem Scope**:
   - For ground state energy alone: use `policy::OnePass` ($\mathcal{O}(N)$ memory).
   - For ground state eigenvector on massive systems ($D > 10^7$): use `policy::TwoPass` (strictly 4 vectors in RAM).
   - For low-lying excited states: use `policy::OnePass_DKGS` (avoids Davidson's preconditioner pole).
3. **Use FTLM Ground-State Shifting**: In thermal simulations, always subtract $E_{\min} = \min_{r, m} \epsilon_m^{(r)}$ from Boltzmann arguments to prevent arithmetic underflow/overflow.
4. **Site Auto-Inference**: Leverage `infer_site_from_basis(basis)` or let `Hamiltonian H(basis, ops)` infer the physical site type automatically.
