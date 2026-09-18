# `qkrylov` Python Package Specification & Developer Guide

This document provides a comprehensive, technically rigorous architectural reference and user guide for the Python bindings of `qkrylov` located in [`bindings/python/`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python). Built on modern [`nanobind`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/src/binding_main.cpp#L1-L17) and [`scikit-build-core`](file:///home/pritipriya/Documents/GitHub/qkrylov/pyproject.toml#L1-L6), `qkrylov` delivers zero-copy NumPy array interoperability, SciPy `LinearOperator` integration, dual-precision arithmetic (`float32`/`complex64` and `float64`/`complex128`), and Kokkos-accelerated matrix-free quantum many-body solvers.

---

## Table of Contents
1. [Architecture & Interoperability](#1-architecture--interoperability)
   - [Nanobind ABI Layer & Capsule Lifecycles](#nanobind-abi-layer--capsule-lifecycles)
   - [Dual Precision Engine (FP64 & FP32)](#dual-precision-engine-fp64--fp32)
   - [Hardware Discovery & Device Trait Management](#hardware-discovery--device-trait-management)
2. [Physical Site Models (`qkrylov.site`)](#2-physical-site-models-qkrylovsite)
   - [Base Class `Site`](#base-class-site)
   - [Spin-1/2 (`SpinHalfSite`)](#spin-12-spinhalfsite)
   - [Arbitrary Spin $S$ (`SpinSSite`)](#arbitrary-spin-s-spinssite)
   - [Spinless Fermions (`FermionSite`)](#spinless-fermions-fermionsite)
   - [Fermi-Hubbard Electrons (`HubbardSite`)](#fermi-hubbard-electrons-hubbardsite)
   - [$t\text{-}J$ Model (`TJSite`)](#t-j-model-tjsite)
3. [Fock Space Bases (`qkrylov.basis`)](#3-fock-space-bases-qkrylovbasis)
   - [Base Class `Basis`](#base-class-basis)
   - [Spin-1/2 Basis (`SpinHalfBasis`)](#spin-12-basis-spinhalfbasis)
   - [Arbitrary Spin $S$ Basis (`SpinSBasis`)](#arbitrary-spin-s-basis-spinsbasis)
   - [Spinless Fermion Basis (`FermionBasis`)](#spinless-fermion-basis-fermionbasis)
   - [Hubbard Electron Basis (`HubbardBasis`)](#hubbard-electron-basis-hubbardbasis)
   - [$t\text{-}J$ Basis (`TJBasis`)](#t-j-basis-tjbasis)
   - [Symmetry Sector Specification](#symmetry-sector-specification)
4. [Symbolic Interactions & Operator DSL (`qkrylov.operators`)](#4-symbolic-interactions--operator-dsl-qkrylovoperators)
   - [Operator Enumeration `Op`](#operator-enumeration-op)
   - [Algebraic Expressions (`LocalOpExpr`, `TermExpr`, `OpSumExpr`)](#algebraic-expressions-localopexpr-termexpr-opsumexpr)
   - [Builder Container `OpSum`](#builder-container-opsum)
5. [Matrix-Free Hamiltonian Operator (`qkrylov.hamiltonian`)](#5-matrix-free-hamiltonian-operator-qkrylovhamiltonian)
   - [Constructor Signatures & Site Auto-Inference](#constructor-signatures--site-auto-inference)
   - [Zero-Copy Evaluation (`apply` and `@`)](#zero-copy-evaluation-apply-and-)
   - [Diagonal Extraction & Device Re-targeting](#diagonal-extraction--device-re-targeting)
   - [SciPy `LinearOperator` & CSR Sparse Conversion](#scipy-linearoperator--csr-sparse-conversion)
6. [Solvers & Algorithms (`qkrylov.solvers`)](#6-solvers--algorithms-qkrylovsolvers)
   - [Abstract Solver Interface `Solver`](#abstract-solver-interface-solver)
   - [Lanczos Ground State (`Lanczos` & `LanczosTwoPass`)](#lanczos-ground-state-lanczos--lanczostwopass)
   - [Davidson Subspace Eigensolver (`Davidson`)](#davidson-subspace-eigensolver-davidson)
   - [Finite Temperature Lanczos Method (`FTLM`)](#finite-temperature-lanczos-method-ftlm)
   - [Pure-State Real-Time Dynamics (`TimeEvolve`)](#pure-state-real-time-dynamics-timeevolve)
   - [Finite-Temperature Dynamical Correlators (`FTLMDynamics`)](#finite-temperature-dynamical-correlators-ftlmdynamics)
   - [Correction Vector Solver (`CorrectionVector`)](#correction-vector-solver-correctionvector)
   - [Result Containers & Protocol Unpacking](#result-containers--protocol-unpacking)
7. [Comprehensive End-to-End Examples](#7-comprehensive-end-to-end-examples)
   - [Example 1: Heisenberg $S=1/2$ Antiferromagnetic Chain](#example-1-heisenberg-s12-antiferromagnetic-chain)
   - [Example 2: Spin-1 Haldane Chain with `SpinSSite` & `SpinSBasis`](#example-2-spin-1-haldane-chain-with-spinssite--spinsbasis)
   - [Example 3: Fermi-Hubbard Model at Half-Filling with $S_z=0$](#example-3-fermi-hubbard-model-at-half-filling-with-s_z0)
   - [Example 4: Dynamical Spectral Function via Continued Fraction](#example-4-dynamical-spectral-function-via-continued-fraction)
   - [Example 5: Correction Vector Conjugate Gradient Spectral Response](#example-5-correction-vector-conjugate-gradient-spectral-response)
   - [Example 6: Finite Temperature Multi-Observable Sweep (FTLM)](#example-6-finite-temperature-multi-observable-sweep-ftlm)
   - [Example 7: SciPy `eigsh` Integration via `aslinearoperator()`](#example-7-scipy-eigsh-integration-via-aslinearoperator)

---

## 1. Architecture & Interoperability

The `qkrylov` Python package couples high-level Python ergonomics with high-throughput C++20 template backends via [`nanobind`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/src/binding_main.cpp#L1-L17).

```
+-------------------------------------------------------------------------------+
|                             Python Client Script                              |
+-------------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------------+
|                       qkrylov Python API Wrapper Layer                         |
|   - qkrylov.operators (OpSum, Sz, Sp, Sm, CdagUp, CUp, etc.)                  |
|   - qkrylov.site      (SpinHalfSite, SpinSSite, FermionSite, Hubbard, TJ)     |
|   - qkrylov.basis     (SpinHalfBasis, SpinSBasis, Fermion, Hubbard, TJ)       |
|   - qkrylov.hamiltonian (MatrixFreeHamiltonian, SciPy LinearOperator, @)      |
|   - qkrylov.solvers   (Lanczos, Davidson, FTLM, ContinuedFraction, CorrVec)  |
+-------------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------------+
|                C++20 Nanobind Extension (_qkrylov_cpp.abi3.so)                 |
|   - src/binding_main.cpp : Common types (Sector, Device, Basis)               |
|   - src/binding_impl.hpp : Template binder for Kokkos execution spaces        |
|   - src/binding_fp32.cpp : Single-precision exports (_FP32)                   |
|   - src/binding_fp64.cpp : Double-precision exports (_FP64)                   |
+-------------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------------+
|                   Core C++20 Header-Only Engine (qkrylov::*)                   |
|   - Kokkos Execution Spaces: Kokkos::OpenMP / Kokkos::Cuda / Kokkos::HIP     |
|   - Matrix-Free Gather SpMV: Kokkos::parallel_for on Fock states              |
+-------------------------------------------------------------------------------+
```

### Nanobind ABI Layer & Capsule Lifecycles

All vector transfers between C++ and Python utilize [`nanobind::capsule`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/src/binding_impl.hpp#L44-L73) to achieve zero-copy semantics without memory leaks:

1. **Host-to-Device / Vector Ingestion**:
   In [`binding_impl.hpp#L42`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/src/binding_impl.hpp#L42), vector parameters are bound as read-only, C-contiguous NumPy views:
   ```cpp
   using CxArray = nb::ndarray<const Complex, nb::shape<-1>, nb::c_contig, nb::device::cpu>;
   ```
   Calling `H.apply(x)` checks that `x` is C-contiguous and matches `H.dimension`.
2. **C++ to NumPy Zero-Copy Output**:
   When C++ allocates a result (e.g. an eigenvector or diagonal), [`vec_to_numpy`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/src/binding_impl.hpp#L44-L58) transfers ownership to a Python `numpy.ndarray`:
   ```cpp
   static nb::ndarray<nb::numpy, Complex, nb::shape<-1>>
   vec_to_numpy(std::vector<Complex>&& v) {
       auto data = std::make_unique<std::vector<Complex>>(std::move(v));
       auto* raw_ptr = data.get();
       nb::capsule owner(raw_ptr, [](void* p) noexcept {
           delete static_cast<std::vector<Complex>*>(p);
       });
       data.release();
       return nb::ndarray<nb::numpy, Complex, nb::shape<-1>>(
           raw_ptr->data(), { raw_ptr->size() }, owner
       );
   }
   ```
   The NumPy array directly references the underlying C++ vector memory; deallocation occurs automatically when Python's garbage collector destroys the array.

### Dual Precision Engine (FP64 & FP32)

Every computational symbol in the C++ layer is explicitly templated and compiled into two distinct precision variants:
* `_FP64`: Double-precision (`qkrylov::fp64`, `Real = double`, `Complex = std::complex<double>`).
* `_FP32`: Single-precision (`qkrylov::fp32`, `Real = float`, `Complex = std::complex<float>`).

The high-level Python layer exposes a unified API and automatically dispatches to the appropriate backend based on the `dtype` parameter:
* `np.float64` / `np.complex128` &rarr; dispatches to `_FP64` symbols.
* `np.float32` / `np.complex64` &rarr; dispatches to `_FP32` symbols.

### Hardware Discovery & Device Trait Management

[`qkrylov.__init__`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/__init__.py#L52-L76) provides system introspection functions to inspect hardware capabilities:

```python
import qkrylov as qk

# Check available GPU backend
backend = qk.find_gpu()  # Returns 'cuda', 'hip', 'sycl', or None
count = qk.gpu_count()   # Number of available accelerators
```

[`Device`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/__init__.py#L65) encapsulates the target execution space:
* `Device("cpu")`: OpenMP, Threads, or Serial execution.
* `Device("cuda")` or `Device("cuda:0")`: NVIDIA GPU acceleration.
* `Device("hip")` or `Device("hip:0")`: AMD ROCm acceleration.
* `Device("sycl")`: Intel oneAPI acceleration.

---

## 2. Physical Site Models (`qkrylov.site`)

Defined in [`bindings/python/qkrylov/site.py`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/site.py), the `Site` hierarchy defines local state transitions, operator matrix elements, and statistics.

### Base Class `Site`
[`qkrylov.site.Site(dtype=np.float32)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/site.py#L5-L10): Base class wrapping native `Site_FP32` or `Site_FP64`.

### Spin-1/2 (`SpinHalfSite`)
[`qkrylov.site.SpinHalfSite(dtype=np.float32)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/site.py#L12-L23) (Alias: `qkrylov.SpinHalfSite`, `qkrylov.site.SpinHalf`):
* Local dimension $d = 2$: $|0\rangle \equiv |\!\downarrow\rangle$, $|1\rangle \equiv |\!\uparrow\rangle$.
* Operators supported:
  * `"Sz"`: $S^z = \frac{1}{2} \sigma^z = \text{diag}(+1/2, -1/2)$
  * `"Sp"`: $S^+ = \sigma^+ = |\!\uparrow\rangle\langle\downarrow\!|$
  * `"Sm"`: $S^- = \sigma^- = |\!\downarrow\rangle\langle\uparrow\!|$
  * `"Sx"`: $S^x = \frac{1}{2} (S^+ + S^-)$
  * `"Sy"`: $S^y = \frac{1}{2i} (S^+ - S^-)$

### Arbitrary Spin $S$ (`SpinSSite`)
[`qkrylov.site.SpinSSite(S=0.5, dtype=np.float32)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/site.py#L24-L47) (Alias: `qkrylov.SpinSSite`, `qkrylov.site.SpinS`):
* Local dimension $d = 2S + 1$:
  * Local basis states $m_s \in \{-S, -S+1, \dots, S-1, S\}$ encoded as integers $k = 0, \dots, 2S$ where $m_s = k - S$.
* Properties:
  * `.spin -> float`: The spin quantum number $S$ ($S > 0$, integer or half-integer).
  * `.dimension_per_site -> int`: $2S + 1$.
* Operators supported:
  * `"Sz"`: $S^z |S, m\rangle = m |S, m\rangle$.
  * `"Sp"`: $S^+ |S, m\rangle = \sqrt{S(S+1) - m(m+1)} |S, m+1\rangle$.
  * `"Sm"`: $S^- |S, m\rangle = \sqrt{S(S+1) - m(m-1)} |S, m-1\rangle$.

### Spinless Fermions (`FermionSite`)
[`qkrylov.site.FermionSite(dtype=np.float32)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/site.py#L48-L56) (Alias: `qkrylov.FermionSite`, `qkrylov.site.Fermion`):
* Local dimension $d = 2$: $|0\rangle$ (empty), $|1\rangle$ (occupied).
* Operators supported:
  * `"C"`: Fermionic annihilation $c_i$. Automatically handles Jordan-Wigner string phase $(-1)^{\sum_{j < i} n_j}$.
  * `"Cdag"`: Fermionic creation $c_i^\dagger$.
  * `"N"`: Particle number operator $n_i = c_i^\dagger c_i$.

### Fermi-Hubbard Electrons (`HubbardSite`)
[`qkrylov.site.HubbardSite(dtype=np.float32)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/site.py#L57-L65) (Alias: `qkrylov.HubbardSite`, `qkrylov.site.Hubbard`):
* Local dimension $d = 4$:
  * $|0\rangle \equiv |0\rangle$ (empty)
  * $|1\rangle \equiv |\!\uparrow\rangle$ (single up)
  * $|2\rangle \equiv |\!\downarrow\rangle$ (single down)
  * $|3\rangle \equiv |\!\uparrow\downarrow\rangle$ (doubly occupied)
* Operators supported:
  * `"CdagUp"`, `"CUp"`: Up-spin creation and annihilation $c_{i\uparrow}^\dagger, c_{i\uparrow}$.
  * `"CdagDn"`, `"CDn"`: Down-spin creation and annihilation $c_{i\downarrow}^\dagger, c_{i\downarrow}$.
  * `"Nup"`: Up-spin occupancy $n_{i\uparrow} = c_{i\uparrow}^\dagger c_{i\uparrow}$.
  * `"Ndn"`: Down-spin occupancy $n_{i\downarrow} = c_{i\downarrow}^\dagger c_{i\downarrow}$.
  * `"Nupdn"`: On-site Hubbard double occupancy $n_{i\uparrow} n_{i\downarrow}$.

### $t\text{-}J$ Model (`TJSite`)
[`qkrylov.site.TJSite(dtype=np.float32)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/site.py#L66-L74) (Alias: `qkrylov.TJSite`, `qkrylov.site.TJ`):
* Local dimension $d = 3$ (Gutzwiller projection: no double occupancy):
  * $|0\rangle \equiv |0\rangle$ (empty / hole)
  * $|1\rangle \equiv |\!\uparrow\rangle$ (single up)
  * $|2\rangle \equiv |\!\downarrow\rangle$ (single down)
* Operators supported:
  * Projected hopping: `"CdagUp"`, `"CUp"`, `"CdagDn"`, `"CDn"`.
  * Spin and density: `"Sz"`, `"Sp"`, `"Sm"`, `"Nup"`, `"Ndn"`.

---

## 3. Fock Space Bases (`qkrylov.basis`)

Defined in [`bindings/python/qkrylov/basis.py`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/basis.py), basis classes generate, store, and binary-search basis states within symmetry sectors.

### Base Class `Basis`
[`qkrylov.basis.Basis`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/basis.py#L5-L20):
* `.size -> int`: Dimension of the restricted or full Hilbert space.
* `.nsites -> int`: Number of sites $N$.

### Spin-1/2 Basis (`SpinHalfBasis`)
[`qkrylov.basis.SpinHalfBasis(N, conserve_sz=False, sz=None, dtype=np.float32)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/basis.py#L50-L87) (Alias: `qkrylov.SpinHalfBasis`, `qkrylov.basis.SpinHalf`):
* Parameters:
  * `N`: Site count $N$.
  * `conserve_sz`: Boolean flag to restrict to a total $S^z$ sector.
  * `sz`: Target $S^z$ sector (e.g. `sz=0` or `sz=1.0`). Passing `sz` automatically implies `conserve_sz=True`.
* Unconstrained dimension: $2^N$.
* Conserved sector dimension: $\binom{N}{N_\uparrow}$ where $N_\uparrow = \frac{N}{2} + S^z$.

### Arbitrary Spin $S$ Basis (`SpinSBasis`)
[`qkrylov.basis.SpinSBasis(N, S=0.5, conserve_sz=False, sz=None, dtype=np.float32)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/basis.py#L89-L158) (Alias: `qkrylov.SpinSBasis`, `qkrylov.basis.SpinS`):
* Parameters:
  * `N`: Site count $N$.
  * `S`: Spin quantum number $S$ ($S \in \{0.5, 1.0, 1.5, 2.0, \dots\}$).
  * `conserve_sz`: Restrict to total $S^z$.
  * `sz`: Target total $S^z = \sum_i m_i$.
* Dimension:
  * Unconstrained: $(2S + 1)^N$.
  * Conserved: Number of weak compositions of $2S \cdot N$ satisfying the total $S^z$ constraint.
* Member Functions:
  * `.state(i: int) -> int`: Retrieve the basis state integer configuration at index $i$.
  * `.index(s: int) -> int`: Binary search for the index of integer configuration $s$ (returns $-1$ if absent).
  * `.contains(s: int) -> bool`: Check if configuration $s$ exists in the sector.

### Spinless Fermion Basis (`FermionBasis`)
[`qkrylov.basis.FermionBasis(N, conserve_n=False, n=0, dtype=np.float32)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/basis.py#L160-L189) (Alias: `qkrylov.FermionBasis`, `qkrylov.basis.Fermion`):
* Parameters:
  * `N`: Total fermionic modes.
  * `conserve_n`: Conserve particle number.
  * `n`: Target particle number $N_p$.

### Hubbard Electron Basis (`HubbardBasis`)
[`qkrylov.basis.HubbardBasis(N, conserve_nup=False, nup=0, conserve_ndn=False, ndn=0, dtype=np.float32)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/basis.py#L191-L233) (Alias: `qkrylov.HubbardBasis`, `qkrylov.basis.Hubbard`):
* Parameters:
  * `N`: Spatial sites.
  * `conserve_nup`: Conserve $N_\uparrow$.
  * `nup`: Number of spin-up electrons.
  * `conserve_ndn`: Conserve $N_\downarrow$.
  * `ndn`: Number of spin-down electrons.
* State layout: 2 bits per site ($\text{bit } 2i$ for up, $\text{bit } 2i+1$ for down).

### $t\text{-}J$ Basis (`TJBasis`)
[`qkrylov.basis.TJBasis(N, conserve_nup=False, nup=0, conserve_ndn=False, ndn=0, dtype=np.float32)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/basis.py#L235-L263) (Alias: `qkrylov.TJBasis`, `qkrylov.basis.TJ`):
* Parameters:
  * `N`: Spatial sites.
  * `nup`: Number of spin-up electrons.
  * `ndn`: Number of spin-down electrons.
* Double occupancy is strictly filtered out during basis generation.

### Symmetry Sector Specification
Internally, [`_build_sector`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/basis.py#L21-L48) constructs a C++ [`Sector`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/src/binding_main.cpp#L19-L30) structure:
* `use_sz` and `sz2`: Stores $2 \cdot S^z$ to represent integer and half-integer sectors without floating-point rounding issues.
* `use_nup` and `nup`: Spin-up particle count.
* `use_ndn` and `ndn`: Spin-down particle count.
* `use_n` and `n`: Total fermion number.
* `use_nb` and `nb`: Total boson number.

---

## 4. Symbolic Interactions & Operator DSL (`qkrylov.operators`)

Defined in [`bindings/python/qkrylov/operators.py`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/operators.py), the operator subsystem provides both a symbolic algebraic DSL and structured tuple addition.

### Operator Enumeration `Op`
[`qkrylov.operators.Op`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/operators.py#L6-L23) prevents spelling errors in operator names:
```python
class Op(str, enum.Enum):
    Sz = "Sz"
    Sp = "Sp"
    Sm = "Sm"
    Sx = "Sx"
    Sy = "Sy"
    CdagUp = "CdagUp"
    CUp = "CUp"
    CdagDn = "CdagDn"
    CDn = "CDn"
    Nup = "Nup"
    Ndn = "Ndn"
    Nupdn = "Nupdn"
    Bdag = "Bdag"
    B = "B"
    N = "N"
```

### Algebraic Expressions (`LocalOpExpr`, `TermExpr`, `OpSumExpr`)
Top-level generator functions instantiate symbolic expressions:
* Spin operators: [`Sz(i)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/operators.py#L111), [`Sp(i)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/operators.py#L112), [`Sm(i)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/operators.py#L113), [`Sx(i)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/operators.py#L114), [`Sy(i)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/operators.py#L115).
* Fermion operators: [`CdagUp(i)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/operators.py#L117), [`CUp(i)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/operators.py#L118), [`CdagDn(i)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/operators.py#L119), [`CDn(i)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/operators.py#L120), [`Nup(i)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/operators.py#L122), [`Ndn(i)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/operators.py#L123), [`Nupdn(i)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/operators.py#L124).
* Boson operators: [`Bdag(i)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/operators.py#L126), [`B(i)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/operators.py#L127), [`N(i)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/operators.py#L128).

These overload `__mul__`, `__rmul__`, `__add__`, and `__sub__` to permit natural mathematical expressions:
```python
# Heisenberg exchange bond
term = 1.0 * qk.Sz(i) * qk.Sz(j) + 0.5 * (qk.Sp(i) * qk.Sm(j) + qk.Sm(i) * qk.Sp(j))

# Hubbard on-site repulsion
hubbard_u = 4.0 * qk.Nupdn(i)
```

### Builder Container `OpSum`
[`qkrylov.OpSum(dtype=np.float32)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/operators.py#L131-L176) compiles interaction terms into C++ memory:

```python
op = qk.OpSum()

# Method 1: In-place addition with algebraic expressions
op += 1.0 * qk.Sz(0) * qk.Sz(1)

# Method 2: In-place addition with tuples (coeff, op1, site1, op2, site2, ...)
op += (0.5, "Sp", 0, "Sm", 1)

# Method 3: Direct method call
op.add_term(0.5, "Sm", 0, "Sp", 1)

# Inspection
print(op.size)   # Number of compiled terms
op.clear()       # Reset terms
```

---

## 5. Matrix-Free Hamiltonian Operator (`qkrylov.hamiltonian`)

Defined in [`bindings/python/qkrylov/hamiltonian.py`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/hamiltonian.py), [`MatrixFreeHamiltonian`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/hamiltonian.py#L8-L256) (Alias: [`Hamiltonian`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/hamiltonian.py#L258)) couples a `Basis`, a `Site`, and an `OpSum` into an executable matrix-free linear operator.

### Constructor Signatures & Site Auto-Inference

The constructor accepts both explicit 3-argument and inferred 2-argument forms:

```python
# 1. Explicit 3-argument invocation:
H = qk.MatrixFreeHamiltonian(basis, site, ops, device="cpu", dtype=np.float64)

# 2. 2-argument invocation (site is inferred automatically from basis):
H = qk.MatrixFreeHamiltonian(basis, ops, device="cpu", dtype=np.float64)
```

**Auto-inference mapping**:
* `SpinHalfBasis` &rarr; `SpinHalfSite(dtype=dtype)`
* `SpinSBasis` &rarr; `SpinSSite(S=basis.spin, dtype=dtype)`
* `FermionBasis` &rarr; `FermionSite(dtype=dtype)`
* `HubbardBasis` &rarr; `HubbardSite(dtype=dtype)`
* `TJBasis` &rarr; `TJSite(dtype=dtype)`

### Zero-Copy Evaluation (`apply` and `@`)

Matrix-vector multiplication $y = H x$ is computed in parallel on device without assembling sparse matrix rows:

```python
# Zero-copy apply method
y = H.apply(x)  # x must be a 1D complex numpy array of size H.dimension

# Python matmul operator (@)
y = H @ x
```

* **Torch Interoperability**: If `x` is a PyTorch `torch.Tensor`, `H @ x` automatically extracts the NumPy buffer, executes the matrix-free SpMV, and returns a `torch.Tensor` on the original device.

### Diagonal Extraction & Device Re-targeting

```python
# Exact diagonal vector of H
diag = H.diagonal()  # Returns 1D numpy array of size H.dimension

# Dimension query
dim = H.dimension    # Equals basis.size

# Retarget to another device or precision
H_gpu = H.to(device="cuda:0", dtype=np.float32)
```

### SciPy `LinearOperator` & CSR Sparse Conversion

[`H.aslinearoperator()`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/hamiltonian.py#L197-L221) wraps the Hamiltonian in a [`scipy.sparse.linalg.LinearOperator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html):
```python
from scipy.sparse.linalg import eigsh

A = H.aslinearoperator()
evals, evecs = eigsh(A, k=3, which='SA')
```

[`H.to_sparse()`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/hamiltonian.py#L223-L255) constructs an explicit [`scipy.sparse.csr_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html) by applying $H$ to all Cartesian unit basis vectors $e_i$ (primarily intended for debugging and verifying small systems).

---

## 6. Solvers & Algorithms (`qkrylov.solvers`)

Defined in [`bindings/python/qkrylov/solvers.py`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/solvers.py), all solvers inherit from [`Solver`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/solvers.py#L37-L48) and provide both Object-Oriented and Functional interfaces.

### Abstract Solver Interface `Solver`

Every solver class satisfies:
```python
class Solver(ABC):
    @abstractmethod
    def solve(self, H: MatrixFreeHamiltonian, *args, **kwargs): ...
    
    def __call__(self, H: MatrixFreeHamiltonian, *args, **kwargs):
        return self.solve(H, *args, **kwargs)
```
Hence, an instantiated solver can be invoked either as `solver.solve(H)` or as a callable `solver(H)`.

---

### Lanczos Ground State (`Lanczos` & `LanczosTwoPass`)

#### 1. Single-Pass Lanczos (`Lanczos`)
[`qkrylov.solvers.Lanczos(maxiter=200, tol=1e-12)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/solvers.py#L50-L84):
* Stores all Krylov basis vectors $v_1, \dots, v_m$ in memory.
* Performs full reorthogonalization to prevent spurious ghost eigenvalues.
* Suitable when system memory $m \times D$ fits comfortably in RAM/VRAM.
* Functional API: [`qkrylov.lanczos_ground_state(H, maxiter=200, tol=1e-12)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/solvers.py#L123-L145).

#### 2. Memory-Frugal Two-Pass Lanczos (`LanczosTwoPass`)
[`qkrylov.solvers.LanczosTwoPass(maxiter=200, tol=1e-12)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/solvers.py#L86-L121):
* **Pass 1**: Computes tridiagonal matrix elements $\alpha_j, \beta_j$ without storing Krylov vectors. Diagonalizes $T_m$ to find ground Ritz eigenvalue $E_0$ and tridiagonal eigenvector $s^{(0)}$.
* **Pass 2**: Re-executes the Lanczos recurrence with the same initial vector $v_1$, accumulating $|\psi_0\rangle = \sum_{j=1}^m s_j^{(0)} |v_j\rangle$ on the fly with strict $\mathcal{O}(3D)$ memory overhead.
* Functional API: [`qkrylov.lanczos_two_pass(H, maxiter=200, tol=1e-12)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/solvers.py#L147-L169).

---

### Davidson Subspace Eigensolver (`Davidson`)

[`qkrylov.solvers.Davidson(n_eig=1, max_subspace=20, tol=1e-8)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/solvers.py#L198-L225):
* Solves for the lowest $k$ eigenpairs simultaneously using diagonal preconditioning:
  $$t_i = \left(\text{diag}(H) - \theta_i I\right)^{-1} r_i$$
* Avoids expanding full Krylov vectors when a good diagonal preconditioner exists.
* Functional API: [`qkrylov.davidson_lowest(H, n_eig=1, max_subspace=20, tol=1e-8)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/solvers.py#L226-L234).

---

### Continued Fraction Dynamics (`ContinuedFraction`)

[`qkrylov.solvers.ContinuedFraction(n_iter=100, phi0=None)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/solvers.py#L256-L283):
* Calculates the tridiagonal coefficients $a_0, a_1, \dots$ and $b_1, b_2, \dots$ representing the resolvent $\langle \phi_0 | (z - H)^{-1} | \phi_0 \rangle$.
* Initial excitation vector: $|\phi_0\rangle = \hat{O} |\psi_0\rangle$.
* Functional APIs:
  * [`qkrylov.continued_fraction_coeffs(H, phi0, n_iter=100)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/solvers.py#L285-L292)
  * [`qkrylov.evaluate_spectral_function(res, omega, E0, eta=0.1)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/solvers.py#L294-L304) evaluates:
    $$A(\omega) = -\frac{1}{\pi} \text{Im} \left[ \frac{\langle \phi_0 | \phi_0 \rangle}{\omega + E_0 - a_0 + i\eta - \frac{b_1^2}{\omega + E_0 - a_1 + i\eta - \dots}} \right]$$

---

### Finite Temperature Lanczos Method (`FTLM`)

[`qkrylov.solvers.FTLM(mode="streamed", beta=1.0, n_random=50, n_steps=100, seed=42)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/solvers.py):
* Uses stochastic trace estimation on the unit sphere $\mathbb{S}^{D-1}$ scaled by $D/R$:
  $$\langle \hat{O} \rangle_\beta = \frac{1}{\bar{Z}(\beta)} \frac{1}{R} \sum_{r=1}^R \sum_{j,k} c_j^{(r)*} \mathcal{O}_{jk}^{(r)} c_k^{(r)} \in \mathbb{C}$$
* Contraction is evaluated as an exact sesquilinear form $c^\dagger \mathcal{O} c$, returning complex expectation values (`np.complex128` or `np.complex64`) for non-Hermitian and complex Hermitian observables.
* Standard errors use covariance-aware linearized ratio variance $\sigma_{\bar{O}}$.

#### Dual Workflows: Streamed vs Cached
1. **Mode 1: Streamed Two-Pass (`mode="streamed"`) [Default]**:
   - Memory-bounded $\mathcal{O}(M^2 N_{\text{obs}})$ footprint (~16 MB).
   - Pass 1 discovers the global minimum shift $E_{\min}$ and caches recurrence coefficients; Pass 2 accumulates on-the-fly for the input `betas` and immediately releases memory.
   ```python
   solver = qk.solvers.FTLM(mode="streamed", n_random=50, n_steps=60)
   sweep = solver.solve(H, betas=[0.1, 0.5, 1.0, 2.0], observables=[O_corr])
   print("Dimension:", sweep.dimension)
   print("Effective samples R_eff:", sweep.effective_samples)
   print("Complex <O>:", sweep.observable_expectations[0])
   ```

2. **Mode 2: Cached Decoupled (`mode="cached"`)**:
   - Stage 1 performs $R$ Krylov walks and stores projected operator matrices in an [`FTLMSamples`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/solvers.py) handle.
   - Stage 2 evaluates thermodynamic properties on arbitrary temperature grids with **zero additional SpMV operations**.
   ```python
   # Stage 1: Sampling
   samples = qk.ftlm_sample(H, observables=[O_corr], n_random=50, n_steps=100, seed=42)

   # Stage 2: Instant re-evaluation
   sweep1 = samples.evaluate_sweep([0.1, 0.5, 1.0, 2.0, 5.0])
   dense_sweep = samples.evaluate_sweep(np.linspace(0.01, 10.0, 200))
   ```

---

### Pure-State Real-Time Dynamics (`TimeEvolve`)

[`qkrylov.solvers.TimeEvolve(n_steps=30)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/solvers.py):
* Propagates an arbitrary initial pure state under unitary time evolution:
  $$|\psi(t)\rangle = e^{-i \hat{H} t} |\psi(0)\rangle$$
* Computes survival probabilities $\mathcal{L}(t) = |\langle \psi(0) | \psi(t) \rangle|^2$ and time-dependent observable expectation values $\langle \hat{O} \rangle(t) = \langle \psi(t) | \hat{O} | \psi(t) \rangle \in \mathbb{C}$.

```python
# OOP API
te_solver = qk.solvers.TimeEvolve(n_steps=30)
rt_res = te_solver.solve(H, psi0=psi0, times=np.linspace(0.0, 10.0, 100), observables=[H, Sz0])

# Functional API
rt_res = qk.time_evolve(H, psi0, times=np.linspace(0.0, 10.0, 100), observables=[H, Sz0], n_steps=30)

print("Survival probabilities:", np.abs(rt_res.survival_probabilities)**2)
print("Observable <H>(t):", rt_res.observable_expectations[0].real)
```

---

### Finite-Temperature Dynamical Correlators (`FTLMDynamics`)

[`qkrylov.solvers.FTLMDynamics(beta=1.0, n_random=50, n_steps=100, seed=42)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/solvers.py):
* Computes unequal-time finite-temperature dynamical correlation functions:
  $$C_{AB}(t; \beta) = \langle \hat{A}(t) \hat{B}(0) \rangle_\beta = \frac{1}{Z(\beta)} \operatorname{Tr}\left( e^{-\beta \hat{H}} e^{i \hat{H} t} \hat{A} e^{-i \hat{H} t} \hat{B} \right)$$
* Employs **dual-Krylov propagation**: builds the branching state $|\chi_0\rangle = \hat{B} |\phi\rangle$ in the full Hilbert space and evolves it in its own Krylov subspace, eliminating projected operator truncation errors and making $C_{AB}(0) = \langle \hat{A} \hat{B} \rangle$ exact to machine precision.

```python
# OOP API
ft_dyn_solver = qk.solvers.FTLMDynamics(beta=1.0, n_random=50, n_steps=100, seed=42)
dyn_res = ft_dyn_solver.solve(H, A=Sz0, B=Sz0, times=np.linspace(0.0, 5.0, 50))

# Functional API
dyn_res = qk.ftlm_dynamics(H, Sz0, Sz0, beta=1.0, times=np.linspace(0.0, 5.0, 50), n_random=50, n_steps=100, seed=42)

print("Correlations C_zz(t):", dyn_res.correlations)
print("Correlation error bars:", dyn_res.correlation_errors)
```

---

### Correction Vector Solver (`CorrectionVector`)

[`qkrylov.solvers.CorrectionVector(e0, omega, eta=0.1, max_iter=500, tol=1e-8, op_psi0=None)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/solvers.py#L444-L504):
* Solves the Hermitian positive-definite squared linear system using Conjugate Gradient:
  $$\left( (H - E_0 - \omega)^2 + \eta^2 \right) |Y\rangle = \eta \hat{O} |\psi_0\rangle$$
* Evaluates the targeted dynamical spectral function without continued-fraction truncation:
  $$S(\omega) = \frac{1}{\pi} \text{Re}\langle \hat{O} \psi_0 | Y \rangle$$
* Functional API: [`qkrylov.correction_vector(H, op_psi0, E0, omega, eta=0.1, max_iter=500, tol=1e-8)`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/solvers.py#L506-L519).

---

### Result Containers & Protocol Unpacking

All result objects support Python unpacking protocols (`__iter__`, `__getitem__`, `__len__`):

| Result Container | Primary Attributes | Tuple Unpacking Syntax |
| :--- | :--- | :--- |
| [`LanczosResult`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/solvers.py#L8-L35) | `energy: float`, `eigenvector: np.ndarray`, `iterations: int`, `converged: bool` | `e0, psi0 = res` |
| [`DavidsonResult`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/solvers.py#L171-L196) | `eigenvalues: np.ndarray`, `eigenvectors: List[np.ndarray]` | `evals, evecs = res` |
| [`DynamicsResult`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/solvers.py#L236-L254) | `alphas: np.ndarray`, `betas: np.ndarray`, `norm_phi0: float` | `alphas, betas, norm0 = res` |
| [`RealTimeResult`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/solvers.py) | `time_grid`, `survival_probabilities: np.ndarray` (complex), `observable_expectations: np.ndarray` (complex) | `times, surv, obs = res` |
| [`FTLMDynamicsResult`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/solvers.py) | `beta`, `time_grid`, `correlations: np.ndarray` (complex), `correlation_errors: np.ndarray` | `beta, times, corr, errs = res` |
| [`FTLMResult`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/solvers.py#L306-L333) | `dimension`, `beta`, `partition_function`, `free_energy`, `internal_energy`, `specific_heat`, `entropy`, `effective_samples`, `observable_expectations`, `observable_errors` | `beta, Z, E, Cv = res` |
| [`FTLMSweepResult`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/solvers.py#L334-L351) | `dimension`, `beta_grid`, `partition_functions`, `free_energies`, `internal_energies`, `specific_heats`, `entropies`, `effective_samples`, `observable_expectations`, `observable_errors` | Accessed via attributes |
| [`FTLMSamples`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/solvers.py#L354) | `num_samples: int`, `.evaluate_sweep(betas)` | Pre-computed Krylov subspace handle |
| [`CorrectionVectorResult`](file:///home/pritipriya/Documents/GitHub/qkrylov/bindings/python/qkrylov/solvers.py#L408-L442) | `correction_vector: np.ndarray`, `spectral_function: float`, `iterations: int`, `converged: bool` | `corr_vec, spec, iters, conv = res` |

---

## 7. Comprehensive End-to-End Examples

### Example 1: Heisenberg $S=1/2$ Antiferromagnetic Chain
Find the ground state of an open 6-site Heisenberg chain in the $S^z = 0$ sector using both Single-Pass and Two-Pass Lanczos:

```python
import numpy as np
import qkrylov as qk

# 1. Define Hilbert space
N = 6
basis = qk.SpinHalfBasis(N, sz=0, dtype=np.float64)
print(f"Hilbert space dimension (Sz=0): {basis.size}")  # C(6, 3) = 20

# 2. Build Hamiltonian terms
ops = qk.OpSum(dtype=np.float64)
for i in range(N - 1):
    ops += 1.0 * qk.Sz(i) * qk.Sz(i + 1)
    ops += 0.5 * (qk.Sp(i) * qk.Sm(i + 1) + qk.Sm(i) * qk.Sp(i + 1))

# 3. Create MatrixFreeHamiltonian (site is inferred automatically)
H = qk.MatrixFreeHamiltonian(basis, ops, dtype=np.float64)

# 4. Solve ground state with Single-Pass Lanczos
sp_solver = qk.solvers.Lanczos(maxiter=100, tol=1e-12)
E0_sp, psi0_sp = sp_solver.solve(H)

# 5. Solve ground state with Two-Pass Lanczos (memory frugal)
tp_solver = qk.solvers.LanczosTwoPass(maxiter=100, tol=1e-12)
E0_tp, psi0_tp = tp_solver(H)

print(f"Single-Pass E0: {E0_sp:.12f}")
print(f"Two-Pass    E0: {E0_tp:.12f}")
assert np.isclose(E0_sp, E0_tp, atol=1e-10)

# Check Rayleigh quotient H @ psi == E0 * psi
rayleigh = np.vdot(psi0_sp, H @ psi0_sp).real / np.linalg.norm(psi0_sp)**2
print(f"Rayleigh quotient: {rayleigh:.12f}")
```

---

### Example 2: Spin-1 Haldane Chain with `SpinSSite` & `SpinSBasis`
Solve for the ground state and low-lying excitations of an $N=4$ Spin-1 Heisenberg chain in the total $S^z = 0$ sector:

```python
import numpy as np
import qkrylov as qk

N = 4
S = 1.0

# 1. Define Spin-1 basis with Sz=0
basis = qk.SpinSBasis(N=N, S=S, sz=0.0, dtype=np.float64)
site = qk.SpinSSite(S=S, dtype=np.float64)
print(f"Spin-1 N=4, Sz=0 dimension: {basis.size}")  # 19

# 2. Build Heisenberg interaction
ops = qk.OpSum(dtype=np.float64)
for i in range(N - 1):
    ops += 1.0 * qk.Sz(i) * qk.Sz(i + 1)
    ops += 0.5 * (qk.Sp(i) * qk.Sm(i + 1) + qk.Sm(i) * qk.Sp(i + 1))

# 3. Construct Hamiltonian
H = qk.MatrixFreeHamiltonian(basis, site, ops, dtype=np.float64)

# 4. Find lowest 3 states with Davidson eigensolver
dav = qk.solvers.Davidson(n_eig=3, max_subspace=15, tol=1e-8)
evals, evecs = dav.solve(H)

print(f"Ground State Energy: {evals[0]:.8f}")
print(f"First Excited State: {evals[1]:.8f}")
print(f"Haldane Gap (OBC):   {evals[1] - evals[0]:.8f}")
```

---

### Example 3: Fermi-Hubbard Model at Half-Filling with $S_z=0$
Simulate a 1D 4-site Fermi-Hubbard chain at half-filling ($N_\uparrow = 2, N_\downarrow = 2$):

```python
import numpy as np
import qkrylov as qk

N = 4
t = 1.0
U = 4.0

# 1. Basis: 2 up and 2 down electrons
basis = qk.HubbardBasis(N=N, nup=2, ndn=2, dtype=np.float64)
print(f"Half-filled Hubbard N=4 dimension: {basis.size}")  # C(4,2) * C(4,2) = 36

# 2. Build Hubbard Hamiltonian
ops = qk.OpSum(dtype=np.float64)

# Hopping terms -t (c_{i,s}^dag c_{i+1,s} + h.c.)
for i in range(N - 1):
    # Up-spin hopping
    ops += -t * qk.CdagUp(i) * qk.CUp(i + 1)
    ops += -t * qk.CdagUp(i + 1) * qk.CUp(i)
    # Down-spin hopping
    ops += -t * qk.CdagDn(i) * qk.CDn(i + 1)
    ops += -t * qk.CdagDn(i + 1) * qk.CDn(i)

# On-site interaction U * n_{i,up} * n_{i,dn}
for i in range(N):
    ops += U * qk.Nupdn(i)

# 3. Hamiltonian & Ground state
H = qk.MatrixFreeHamiltonian(basis, ops, dtype=np.float64)
res = qk.lanczos_ground_state(H, maxiter=100, tol=1e-12)
print(f"Half-filled Hubbard Ground State Energy (U={U}): {res.energy:.10f}")
```

---

### Example 4: Dynamical Spectral Function via Continued Fraction
Compute the local dynamical spin structure factor $S_{zz}(\omega) = -\frac{1}{\pi} \text{Im} \langle \psi_0 | S^z_0 \frac{1}{\omega + E_0 - H + i\eta} S^z_0 | \psi_0 \rangle$:

```python
import numpy as np
import qkrylov as qk

# 1. System setup
N = 6
basis = qk.SpinHalfBasis(N, sz=0, dtype=np.float64)
ops = qk.OpSum(dtype=np.float64)
for i in range(N - 1):
    ops += 1.0 * qk.Sz(i) * qk.Sz(i + 1) + 0.5 * (qk.Sp(i) * qk.Sm(i + 1) + qk.Sm(i) * qk.Sp(i + 1))
H = qk.MatrixFreeHamiltonian(basis, ops, dtype=np.float64)

# 2. Ground state
E0, psi0 = qk.lanczos_ground_state(H, maxiter=100, tol=1e-12)

# 3. Excitation: |phi0> = Sz(0) |psi0>
op_sz0 = qk.OpSum(dtype=np.float64)
op_sz0 += 1.0 * qk.Sz(0)
H_sz0 = qk.MatrixFreeHamiltonian(basis, op_sz0, dtype=np.float64)
phi0 = H_sz0 @ psi0

# 4. Lanczos Continued Fraction
dyn_res = qk.continued_fraction_coeffs(H, phi0, n_iter=40)
print(f"Norm of |phi0>: {dyn_res.norm_phi0:.6f}")

# 5. Evaluate spectral function over frequency grid
omegas = np.linspace(-1.0, 4.0, 200)
eta = 0.05
spec = [qk.evaluate_spectral_function(dyn_res, omega=w, E0=E0, eta=eta) for w in omegas]

print(f"Peak spectral weight around: {omegas[np.argmax(spec)]:.3f}")
```

---

### Example 5: Correction Vector Conjugate Gradient Spectral Response
Compute exact point-by-point spectral response with Conjugate Gradient:

```python
import numpy as np
import qkrylov as qk

# 1. System setup & ground state
N = 4
basis = qk.SpinHalfBasis(N, sz=0, dtype=np.float64)
ops = qk.OpSum(dtype=np.float64)
for i in range(N - 1):
    ops += 1.0 * qk.Sz(i) * qk.Sz(i + 1) + 0.5 * (qk.Sp(i) * qk.Sm(i + 1) + qk.Sm(i) * qk.Sp(i + 1))
H = qk.MatrixFreeHamiltonian(basis, ops, dtype=np.float64)
E0, psi0 = qk.lanczos_ground_state(H)

# 2. Excitation operator |phi0> = S^z_0 |psi0>
op_sz0 = qk.OpSum(dtype=np.float64)
op_sz0 += 1.0 * qk.Sz(0)
H_sz0 = qk.MatrixFreeHamiltonian(basis, op_sz0, dtype=np.float64)
op_psi0 = H_sz0 @ psi0

# 3. Targeted frequency solve via Correction Vector
omega_target = 1.2
eta = 0.1
cv_solver = qk.solvers.CorrectionVector(e0=E0, omega=omega_target, eta=eta, max_iter=300, tol=1e-8)
cv_res = cv_solver.solve(H, op_psi0)

print(f"S(omega={omega_target}): {cv_res.spectral_function:.10e}")
print(f"CG Iterations:        {cv_res.iterations} (Converged: {cv_res.converged})")
```

---

### Example 6: Finite Temperature Multi-Observable Sweep (FTLM)
Compute thermodynamic equations of state (Partition function $Z$, Free energy $F$, Energy $E$, Specific heat $C_v$, Entropy $S$) and measure $\langle H \rangle$:

```python
import numpy as np
import qkrylov as qk

# 1. System setup
N = 4
basis = qk.SpinHalfBasis(N, sz=0, dtype=np.float64)
ops = qk.OpSum(dtype=np.float64)
for i in range(N - 1):
    ops += 1.0 * qk.Sz(i) * qk.Sz(i + 1) + 0.5 * (qk.Sp(i) * qk.Sm(i + 1) + qk.Sm(i) * qk.Sp(i + 1))
H = qk.MatrixFreeHamiltonian(basis, ops, dtype=np.float64)

# 2. Decoupled 2-Stage FTLM:
# Stage 1: Generate Krylov subspace samples & project observables (heavy SpMV occurs here)
samples = qk.ftlm_sample(
    H,
    observables=[H],
    n_random=30,
    n_steps=40,
    seed=1234
)
print(f"Generated {samples.num_samples} Krylov subspace samples.")

# Stage 2: Evaluate on temperature grid (zero additional SpMV operations)
betas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
ftlm_res = samples.evaluate_sweep(betas)

print(f"{'Beta':<8}{'Internal Energy':<18}{'Specific Heat':<16}{'<H> Obs':<16}")
for i, b in enumerate(ftlm_res.beta_grid):
    E = ftlm_res.internal_energies[i]
    Cv = ftlm_res.specific_heats[i]
    obs_H = ftlm_res.observable_expectations[0][i]
    print(f"{b:<8.2f}{E:<18.6f}{Cv:<16.6f}{obs_H:<16.6f}")

# Re-evaluate instantaneously on a dense 100-point grid without re-running Lanczos
dense_grid = np.linspace(0.05, 10.0, 100)
dense_res = samples.evaluate_sweep(dense_grid)
print(f"Instantly evaluated {len(dense_res.beta_grid)} temperature points on the dense grid!")
```

---

### Example 7: SciPy `eigsh` Integration via `aslinearoperator()`
Pass the matrix-free Hamiltonian directly to SciPy:

```python
import numpy as np
import scipy.sparse.linalg as sla
import qkrylov as qk

# 1. Setup Hamiltonian
basis = qk.SpinHalfBasis(N=8, sz=0, dtype=np.float64)
ops = qk.OpSum(dtype=np.float64)
for i in range(7):
    ops += 1.0 * qk.Sz(i) * qk.Sz(i + 1) + 0.5 * (qk.Sp(i) * qk.Sm(i + 1) + qk.Sm(i) * qk.Sp(i + 1))
H = qk.MatrixFreeHamiltonian(basis, ops, dtype=np.float64)

# 2. Convert to SciPy LinearOperator
A = H.aslinearoperator()
print(f"Operator shape: {A.shape}, dtype: {A.dtype}")

# 3. Compute 4 lowest eigenvalues using SciPy's ARPACK wrapper
evals, evecs = sla.eigsh(A, k=4, which="SA", tol=1e-10)

print(f"Lowest 4 Eigenvalues via SciPy eigsh:")
for idx, val in enumerate(evals):
    print(f"  Level {idx}: {val:.10f}")
```
