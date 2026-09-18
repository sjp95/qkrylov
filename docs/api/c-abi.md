# C ABI Reference & Export Guide for `qkrylov` (`libqkrylov.so`)

The `qkrylov` C ABI defines a flat, binary-stable `extern "C"` application binary interface for foreign function interfacing (FFI) with Julia, Python, Rust, Fortran, and C. The C ABI is declared in [`include/qkrylov/c_api.h`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/c_api.h#L1-L640) and implemented across [`src/c_api.cpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/src/c_api.cpp), [`src/c_api_fp32.cpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/src/c_api_fp32.cpp), [`src/c_api_fp64.cpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/src/c_api_fp64.cpp), and [`src/c_api_impl.hpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/src/c_api_impl.hpp).

---

## Table of Contents

1. [Architectural Principles & ABI Design](#1-architectural-principles--abi-design)
2. [Return Codes & Thread-Local Diagnostics](#2-return-codes--thread-local-diagnostics)
3. [Opaque Handles & Descriptor Model](#3-opaque-handles--descriptor-model)
4. [Quantum Number Sector API](#4-quantum-number-sector-api)
5. [Hilbert Space Basis API & Reflection](#5-hilbert-space-basis-api--reflection)
6. [Physical Site API, Reflection & Local Action](#6-physical-site-api-reflection--local-action)
7. [Operator Sum (`OpSum`) API & Introspection](#7-operator-sum-opsum-api--introspection)
8. [Hardware Device & Platform Queries](#8-hardware-device--platform-queries)
9. [Matrix-Free Hamiltonian API](#9-matrix-free-hamiltonian-api)
10. [Numerical Solvers & Eigensystem Computations](#10-numerical-solvers--eigensystem-computations)
    - [10.1 Lanczos Ground State Solvers (Single-Pass & Two-Pass)](#101-lanczos-ground-state-solvers-single-pass--two-pass)
    - [10.2 Multi-State Lanczos with DGKS Reorthogonalization](#102-multi-state-lanczos-with-dgks-reorthogonalization)
    - [10.3 Block Davidson Solver](#103-block-davidson-solver)
    - [10.4 Dynamical Continued Fraction Spectroscopy](#104-dynamical-continued-fraction-spectroscopy)
    - [10.5 Finite-Temperature Lanczos Method (Dual Workflows: Streamed & Cached)](#105-finite-temperature-lanczos-method-dual-workflows-streamed--cached)
    - [10.6 Pure-State Real-Time Dynamics (time_evolve)](#106-pure-state-real-time-dynamics-time_evolve)
    - [10.7 Finite-Temperature Dynamical Correlators (ftlm_dynamics)](#107-finite-temperature-dynamical-correlators-ftlm_dynamics)
    - [10.8 Correction Vector Method](#108-correction-vector-method)
11. [Parallel Vector BLAS-1 API (Host & Device-Resident)](#11-parallel-vector-blas-1-api-host--device-resident)
12. [Complete C Program Usage Example](#12-complete-c-program-usage-example)

---

## 1. Architectural Principles & ABI Design

### 1.1 Dual-Precision Endpoint Pairings
Every computational entry point is exposed with explicit precision suffixes:
* **`_fp64`**: 64-bit IEEE 754 precision (`double`, `ComplexF64`). Used for CPU simulations and datacenter GPUs requiring strict physical convergence down to $\approx 10^{-12}\text{--}10^{-15}$.
* **`_fp32`**: 32-bit precision (`float`, `ComplexF32`). Used for consumer GPUs (GeForce RTX) to bypass FP64 compute throttling and halve VRAM usage so twice as large Hilbert spaces fit in device memory.
* **Unsuffixed Aliases**: Functions without `_fp32` or `_fp64` (e.g. `qkrylov_lanczos_ground_state`) alias directly to the double-precision (`_fp64`) implementation.

### 1.2 Zero-Flag Solver Architecture
To prevent runtime branching and ambiguous boolean configurations across FFI boundaries, solvers expose dedicated function names:
* **Single-Pass Lanczos**: `qkrylov_lanczos_ground_state_fp64` (retains full basis in RAM or solves scalar energy).
* **Two-Pass Lanczos**: `qkrylov_lanczos_two_pass_ground_state_fp64` (strictly $\mathcal{O}(N)$ RAM via seed replay).

### 1.3 Strict Exception Barrier
C++ exceptions cannot cross C ABI boundaries without causing undefined behavior or process termination in foreign runtimes (like Julia or Python). Every function in `libqkrylov.so` catches `std::exception` and unexpected errors, writing diagnostic details into thread-local storage and returning an integer error code.

### 1.4 Zero-Copy Buffer Layouts
Functions accepting complex vectors expect standard contiguous interleaved arrays of IEEE floating-point numbers:
```text
[Re(v_0), Im(v_0), Re(v_1), Im(v_1), ..., Re(v_{D-1}), Im(v_{D-1})]
```
Length in scalar elements is $2 \times D$.

---

## 2. Return Codes & Thread-Local Diagnostics

### 2.1 Return Error Codes
Defined in [`include/qkrylov/c_api.h`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/c_api.h#L22-L28):
```c
#define QKRYLOV_SUCCESS                0
#define QKRYLOV_ERROR_INVALID_ARG     -1
#define QKRYLOV_ERROR_EXCEPTION       -2

#define QKRYLOV_PRECISION_FP32         0
#define QKRYLOV_PRECISION_FP64         1
```

### 2.2 Error Diagnostic API
Declared in [`include/qkrylov/c_api.h`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/c_api.h#L35-L38):

```c
// Returns last error caught across C ABI barrier on current thread ("" if none)
const char* qkrylov_get_last_error_message(void);

// Clears the thread-local error string
void qkrylov_clear_last_error(void);
```

Diagnostic messages are completely thread-isolated: errors thrown on one thread will not leak to or overwrite errors on another calling thread.

---

## 3. Opaque Handles & Descriptor Model

Defined in [`include/qkrylov/c_api.h`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/c_api.h#L43-L56) and [`src/c_api_internal.hpp`](file:///home/pritipriya/Documents/GitHub/qkrylov/src/c_api_internal.hpp#L13-L68):

```c
typedef struct qkrylov_sector_t*        qkrylov_sector_h;
typedef struct qkrylov_basis_t*         qkrylov_basis_h;
typedef struct qkrylov_site_t*          qkrylov_site_h;
typedef struct qkrylov_opsum_t*         qkrylov_opsum_h;
typedef struct qkrylov_hamiltonian_t*   qkrylov_hamiltonian_h;
typedef struct qkrylov_device_vector_t* qkrylov_device_vector_h;
typedef struct qkrylov_ftlm_samples_t*  qkrylov_ftlm_samples_h;
typedef qkrylov_ftlm_samples_h          qkrylov_ftlm_samples_fp64_h;
typedef qkrylov_ftlm_samples_h          qkrylov_ftlm_samples_fp32_h;
```

**Descriptor Reuse**: `qkrylov_basis_h`, `qkrylov_site_h`, and `qkrylov_opsum_h` are lightweight configuration descriptors. The same handles can be used to construct multiple Hamiltonians (e.g. one in FP32 on GPU and one in FP64 on CPU) without re-specifying terms. `qkrylov_ftlm_samples_h` preserves Krylov subspace samples across multiple temperature sweeps.

---

## 4. Quantum Number Sector API

Declared in [`include/qkrylov/c_api.h`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/c_api.h#L61-L70):

```c
// Allocation and lifecycle
qkrylov_sector_h qkrylov_sector_create(void);
void             qkrylov_sector_destroy(qkrylov_sector_h sector);

// Set symmetry constraints
int qkrylov_sector_set_sz(qkrylov_sector_h sector, int sz2);
int qkrylov_sector_set_hubbard_particles(qkrylov_sector_h sector, int nup, int ndn);
int qkrylov_sector_set_n(qkrylov_sector_h sector, int n);
int qkrylov_sector_set_nb(qkrylov_sector_h sector, int nb);

// Query symmetry constraints
int qkrylov_sector_get_sz(qkrylov_sector_h sector, int* sz2_out, int* active_out);
int qkrylov_sector_get_hubbard_particles(qkrylov_sector_h sector, int* nup_out, int* ndn_out, int* active_out);
int qkrylov_sector_get_n(qkrylov_sector_h sector, int* n_out, int* active_out);
int qkrylov_sector_get_nb(qkrylov_sector_h sector, int* nb_out, int* active_out);
```

* `sz2`: Represents $2 \times S^z$. For $S^z = 0$, pass `0`; for $S^z = +1/2$, pass `1`; for $S^z = -1$, pass `-2`.

---

## 5. Hilbert Space Basis API & Reflection

Declared in [`include/qkrylov/c_api.h`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/c_api.h#L75-L99):

```c
typedef enum {
    QKRYLOV_BASIS_SPIN_HALF = 0,
    QKRYLOV_BASIS_SPIN_S    = 1,
    QKRYLOV_BASIS_FERMION   = 2,
    QKRYLOV_BASIS_HUBBARD   = 3,
    QKRYLOV_BASIS_TJ        = 4
} qkrylov_basis_type_t;

// Constructors
qkrylov_basis_h qkrylov_spinhalf_basis_create(int num_sites, qkrylov_sector_h sector);
qkrylov_basis_h qkrylov_basis_create_spin_s(int N, double S, const qkrylov_sector_t* sector);
qkrylov_basis_h qkrylov_fermion_basis_create(int num_sites, qkrylov_sector_h sector);
qkrylov_basis_h qkrylov_hubbard_basis_create(int num_sites, qkrylov_sector_h sector);
qkrylov_basis_h qkrylov_tj_basis_create(int num_sites, qkrylov_sector_h sector);
void            qkrylov_basis_destroy(qkrylov_basis_h basis);

// Hilbert space state mapping
uint64_t qkrylov_basis_dimension(qkrylov_basis_h basis);
int      qkrylov_basis_nsites(qkrylov_basis_h basis);
uint64_t qkrylov_basis_state(qkrylov_basis_h basis, uint64_t index);
int64_t  qkrylov_basis_index(qkrylov_basis_h basis, uint64_t state_bitstring);
int      qkrylov_basis_contains(qkrylov_basis_h basis, uint64_t state_bitstring);

// Introspection & Reflection
int              qkrylov_basis_get_type(qkrylov_basis_h basis, qkrylov_basis_type_t* type_out);
int              qkrylov_basis_get_spin(qkrylov_basis_h basis, double* spin_out);
int              qkrylov_basis_get_dimension_per_site(qkrylov_basis_h basis, int* d_out);
qkrylov_sector_h qkrylov_basis_get_sector(qkrylov_basis_h basis);
```

---

## 6. Physical Site API, Reflection & Local Action

Declared in [`include/qkrylov/c_api.h`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/c_api.h#L104-L132):

```c
typedef enum {
    QKRYLOV_SITE_SPIN_HALF = 0,
    QKRYLOV_SITE_SPIN_S    = 1,
    QKRYLOV_SITE_FERMION   = 2,
    QKRYLOV_SITE_HUBBARD   = 3,
    QKRYLOV_SITE_TJ        = 4
} qkrylov_site_type_t;

typedef struct {
    int      valid;             // 1 if transition non-zero, 0 if annihilated
    uint64_t new_state;         // Target Fock bitstring
    double   matrix_element_re; // Re<s'|O|s>
    double   matrix_element_im; // Im<s'|O|s>
} qkrylov_local_action_t;

// Constructors
qkrylov_site_h qkrylov_spinhalf_site_create(void);
qkrylov_site_h qkrylov_site_create_spin_s(double S);
qkrylov_site_h qkrylov_fermion_site_create(void);
qkrylov_site_h qkrylov_hubbard_site_create(void);
qkrylov_site_h qkrylov_tj_site_create(void);
void           qkrylov_site_destroy(qkrylov_site_h site);

// Direct local action evaluation
int qkrylov_site_apply(qkrylov_site_h site, const char* op, int site_idx,
                       uint64_t state, qkrylov_local_action_t* action_out);

// Reflection
int qkrylov_site_get_type(qkrylov_site_h site, qkrylov_site_type_t* type_out);
int qkrylov_site_get_spin(qkrylov_site_h site, double* spin_out);
int qkrylov_site_get_dimension_per_site(qkrylov_site_h site, int* d_out);
```

---

## 7. Operator Sum (`OpSum`) API & Introspection

Declared in [`include/qkrylov/c_api.h`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/c_api.h#L136-L174):

```c
qkrylov_opsum_h qkrylov_opsum_create(void);
void            qkrylov_opsum_destroy(qkrylov_opsum_h opsum);
int             qkrylov_opsum_clear(qkrylov_opsum_h opsum);
int             qkrylov_opsum_size(qkrylov_opsum_h opsum, int* size_out);

// Single-Precision (FP32) Term Adders
int qkrylov_opsum_add_term_1body_fp32(qkrylov_opsum_h opsum, float coeff_re, float coeff_im, const char* op1, int site1);
int qkrylov_opsum_add_term_2body_fp32(qkrylov_opsum_h opsum, float coeff_re, float coeff_im, const char* op1, int site1, const char* op2, int site2);
int qkrylov_opsum_add_term_nbody_fp32(qkrylov_opsum_h opsum, float coeff_re, float coeff_im, int n_factors, const char** ops, const int* sites);

// Double-Precision (FP64) Term Adders (also default aliases)
int qkrylov_opsum_add_term_1body_fp64(qkrylov_opsum_h opsum, double coeff_re, double coeff_im, const char* op1, int site1);
int qkrylov_opsum_add_term_2body_fp64(qkrylov_opsum_h opsum, double coeff_re, double coeff_im, const char* op1, int site1, const char* op2, int site2);
int qkrylov_opsum_add_term_nbody_fp64(qkrylov_opsum_h opsum, double coeff_re, double coeff_im, int n_factors, const char** ops, const int* sites);

// OpSum Term Introspection
int qkrylov_opsum_get_term_info(qkrylov_opsum_h opsum, int term_idx, double* coeff_re_out, double* coeff_im_out, int* num_factors_out);
int qkrylov_opsum_get_factor(qkrylov_opsum_h opsum, int term_idx, int factor_idx, char* op_buf, int op_buf_len, int* site_out);
```

---

## 8. Hardware Device & Platform Queries

Declared in [`include/qkrylov/c_api.h`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/c_api.h#L178-L182):

```c
int         qkrylov_is_gpu_build(void);           // 1 if CUDA/HIP/SYCL enabled, 0 otherwise
const char* qkrylov_find_gpu(void);               // Returns "cuda", "hip", "sycl", or "none"
int         qkrylov_gpu_count(void);              // Number of GPUs detected
int         qkrylov_initialize_device(const char* device_str); // e.g. "cuda:0", "cpu"
```

---

## 9. Matrix-Free Hamiltonian API

Declared in [`include/qkrylov/c_api.h`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/c_api.h#L186-L255):

### 9.1 Lifecycle & Properties
```c
void     qkrylov_hamiltonian_destroy(qkrylov_hamiltonian_h h);
uint64_t qkrylov_hamiltonian_dimension(qkrylov_hamiltonian_h h);
int      qkrylov_hamiltonian_precision(qkrylov_hamiltonian_h h); // 0 = FP32, 1 = FP64
```

### 9.2 Construction (FP32 & FP64)
```c
// FP32
qkrylov_hamiltonian_h qkrylov_hamiltonian_create_fp32(qkrylov_basis_h basis, qkrylov_site_h site, qkrylov_opsum_h opsum);
qkrylov_hamiltonian_h qkrylov_hamiltonian_create_device_fp32(qkrylov_basis_h basis, qkrylov_site_h site, qkrylov_opsum_h opsum, const char* device_str);

// FP64 (and default aliases)
qkrylov_hamiltonian_h qkrylov_hamiltonian_create_fp64(qkrylov_basis_h basis, qkrylov_site_h site, qkrylov_opsum_h opsum);
qkrylov_hamiltonian_h qkrylov_hamiltonian_create_device_fp64(qkrylov_basis_h basis, qkrylov_site_h site, qkrylov_opsum_h opsum, const char* device_str);
```

### 9.3 Action Evaluation ($y = \hat{H}x$)
```c
// Interleaved complex arrays: [re0, im0, re1, im1, ...]
int qkrylov_hamiltonian_apply_complex_fp32(qkrylov_hamiltonian_h h, const float* x_complex, float* y_complex);
int qkrylov_hamiltonian_apply_complex_fp64(qkrylov_hamiltonian_h h, const double* x_complex, double* y_complex);

// Split real and imaginary arrays
int qkrylov_hamiltonian_apply_fp32(qkrylov_hamiltonian_h h, const float* x_re, const float* x_im, float* y_re, float* y_im);
int qkrylov_hamiltonian_apply_fp64(qkrylov_hamiltonian_h h, const double* x_re, const double* x_im, double* y_re, double* y_im);

// Zero-copy device vector multiply
int qkrylov_hamiltonian_apply_device_fp32(qkrylov_hamiltonian_h h, const qkrylov_device_vector_h x_dev, qkrylov_device_vector_h y_dev);
int qkrylov_hamiltonian_apply_device_fp64(qkrylov_hamiltonian_h h, const qkrylov_device_vector_h x_dev, qkrylov_device_vector_h y_dev);
```

### 9.4 Diagonal Extraction ($D_{jj} = \langle j | \hat{H} | j \rangle$)
```c
int qkrylov_hamiltonian_diagonal_fp32(qkrylov_hamiltonian_h h, float* diag_out);
int qkrylov_hamiltonian_diagonal_fp64(qkrylov_hamiltonian_h h, double* diag_out);
int qkrylov_hamiltonian_diagonal_device_fp32(qkrylov_hamiltonian_h h, qkrylov_device_vector_h diag_out);
int qkrylov_hamiltonian_diagonal_device_fp64(qkrylov_hamiltonian_h h, qkrylov_device_vector_h diag_out);
```

---

## 10. Numerical Solvers & Eigensystem Computations

Declared in [`include/qkrylov/c_api.h`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/c_api.h#L259-L495):

### 10.1 Lanczos Ground State Solvers (Single-Pass & Two-Pass)

```c
typedef struct { float  energy; int iterations; int converged; } qkrylov_lanczos_result_fp32_t;
typedef struct { double energy; int iterations; int converged; } qkrylov_lanczos_result_fp64_t;
```

#### Single-Pass Lanczos
```c
// Energy only
int qkrylov_lanczos_ground_state_fp32(qkrylov_hamiltonian_h h, int maxiter, float tol, qkrylov_lanczos_result_fp32_t* result);
int qkrylov_lanczos_ground_state_fp64(qkrylov_hamiltonian_h h, int maxiter, double tol, qkrylov_lanczos_result_fp64_t* result);

// Energy + Eigenvector return
int qkrylov_lanczos_ground_state_complex_fp32(qkrylov_hamiltonian_h h, int maxiter, float tol, qkrylov_lanczos_result_fp32_t* result, float* eigenvector_complex);
int qkrylov_lanczos_ground_state_complex_fp64(qkrylov_hamiltonian_h h, int maxiter, double tol, qkrylov_lanczos_result_fp64_t* result, double* eigenvector_complex);
```

#### Two-Pass Lanczos (Low-Memory Seed Replay)
Requires strictly 4 device vectors in RAM:
```c
int qkrylov_lanczos_two_pass_ground_state_fp32(qkrylov_hamiltonian_h h, int maxiter, float tol, qkrylov_lanczos_result_fp32_t* result);
int qkrylov_lanczos_two_pass_ground_state_fp64(qkrylov_hamiltonian_h h, int maxiter, double tol, qkrylov_lanczos_result_fp64_t* result);

int qkrylov_lanczos_two_pass_ground_state_complex_fp32(qkrylov_hamiltonian_h h, int maxiter, float tol, qkrylov_lanczos_result_fp32_t* result, float* eigenvector_complex);
int qkrylov_lanczos_two_pass_ground_state_complex_fp64(qkrylov_hamiltonian_h h, int maxiter, double tol, qkrylov_lanczos_result_fp64_t* result, double* eigenvector_complex);
```

---

### 10.2 Multi-State Lanczos with DGKS Reorthogonalization

Computes $n_{\text{eig}}$ lowest states using 2-pass DGKS reorthogonalization:

```c
typedef struct { int iterations; int converged; } qkrylov_lanczos_lowest_result_c_t;

int qkrylov_lanczos_lowest_complex_fp32(
    qkrylov_hamiltonian_h h,
    int n_eig, int maxiter, float tol,
    float* eigenvalues_out,            // size n_eig
    float* eigenvectors_complex_out,   // size 2 * n_eig * D
    qkrylov_lanczos_lowest_result_c_t* result_info,
    const float* initial_vector_complex // optional (may be NULL)
);

int qkrylov_lanczos_lowest_complex_fp64(
    qkrylov_hamiltonian_h h,
    int n_eig, int maxiter, double tol,
    double* eigenvalues_out,
    double* eigenvectors_complex_out,
    qkrylov_lanczos_lowest_result_c_t* result_info,
    const double* initial_vector_complex
);
```

---

### 10.3 Block Davidson Solver

```c
typedef struct { int iterations; int converged; } qkrylov_davidson_result_c_t;

int qkrylov_davidson_lowest_complex_fp32(
    qkrylov_hamiltonian_h h,
    int n_eig, int max_subspace, float tol,
    float* eigenvalues_out,
    float* eigenvectors_complex_out,
    qkrylov_davidson_result_c_t* result_info
);

int qkrylov_davidson_lowest_complex_fp64(
    qkrylov_hamiltonian_h h,
    int n_eig, int max_subspace, double tol,
    double* eigenvalues_out,
    double* eigenvectors_complex_out,
    qkrylov_davidson_result_c_t* result_info
);
```

---

### 10.4 Dynamical Continued Fraction Spectroscopy

```c
int qkrylov_continued_fraction_coeffs_complex_fp32(
    qkrylov_hamiltonian_h h,
    const float* phi0_complex, int n_iter,
    float* alphas_out, float* betas_out, float* norm_phi0_out, int* num_coeffs_out
);

int qkrylov_continued_fraction_coeffs_complex_fp64(
    qkrylov_hamiltonian_h h,
    const double* phi0_complex, int n_iter,
    double* alphas_out, double* betas_out, double* norm_phi0_out, int* num_coeffs_out
);

float  qkrylov_evaluate_spectral_function_fp32(const float* alphas, const float* betas, size_t n, float norm_phi0, float omega, float E0, float eta);
double qkrylov_evaluate_spectral_function_fp64(const double* alphas, const double* betas, size_t n, double norm_phi0, double omega, double E0, double eta);
```

### 10.5 Finite-Temperature Lanczos Method (Dual Workflows: Streamed & Cached)

#### 10.5.1 Result Structs
```c
typedef struct {
    float beta;
    float partition_function;
    float free_energy;
    float internal_energy;
    float specific_heat;
    float entropy;
} qkrylov_ftlm_result_fp32_t;

typedef struct {
    double beta;
    double partition_function;
    double free_energy;
    double internal_energy;
    double specific_heat;
    double entropy;
} qkrylov_ftlm_result_fp64_t;

typedef struct {
    int num_betas;
    int num_observables;
    int64_t dimension;
    const float* beta_grid;
    const float* partition_functions;
    const float* free_energies;
    const float* internal_energies;
    const float* specific_heats;
    const float* entropies;
    const float* effective_samples;          /* R_eff(beta) diagnostic array */
    const float* observable_expectations_re; /* Row-major: num_observables x num_betas */
    const float* observable_expectations_im; /* Row-major: num_observables x num_betas */
    const float* observable_expectations;    /* Points to observable_expectations_re */
    const float* observable_errors;          /* Row-major: num_observables x num_betas */
} qkrylov_ftlm_sweep_result_fp32_t;

typedef struct {
    int num_betas;
    int num_observables;
    int64_t dimension;
    const double* beta_grid;
    const double* partition_functions;
    const double* free_energies;
    const double* internal_energies;
    const double* specific_heats;
    const double* entropies;
    const double* effective_samples;          /* R_eff(beta) diagnostic array */
    const double* observable_expectations_re; /* Row-major: num_observables x num_betas */
    const double* observable_expectations_im; /* Row-major: num_observables x num_betas */
    const double* observable_expectations;    /* Points to observable_expectations_re */
    const double* observable_errors;          /* Row-major: num_observables x num_betas */
} qkrylov_ftlm_sweep_result_fp64_t;
```

#### 10.5.2 Dual FTLM Workflows

`qkrylov` provides two first-class FTLM execution modes:

1. **Streamed Two-Pass Mode (`qkrylov_ftlm_sweep_streamed_fp64/fp32`)**:
   - **Peak Host RAM**: Bounded strictly to $\mathcal{O}(M^2 N_{\text{obs}})$ (~16 MB).
   - **Execution**: Pass 1 runs Lanczos on $\hat{H}$ only, discovers the global shift $E_{\min}$, and caches recurrence coefficients. Pass 2 streams through samples $r = 1 \dots R$, projects observables into $M \times M$ matrices, accumulates Boltzmann sums for the input `beta_grid` on the fly, and **immediately frees** basis vectors and operator matrices.
   - **Recommended for**: Large Hilbert spaces, large numbers of observables ($N_{\text{obs}} \ge 10$), and production runs.

2. **Cached Mode (`qkrylov_ftlm_sample_fp64/fp32` + `qkrylov_ftlm_evaluate_sweep_fp64/fp32`)**:
   - **Peak Host RAM**: $\mathcal{O}(R \cdot M^2 N_{\text{obs}})$ (~8 GB).
   - **Execution**: Stage 1 performs $R$ Krylov walks and operator projections, storing the sample collection in an opaque `qkrylov_ftlm_samples_h` handle. Stage 2 evaluates thermodynamic properties on an arbitrary `beta_grid` with **zero additional SpMV operations**.
   - **Recommended for**: Interactive exploration and parameter sweeps across arbitrary temperature grids.

#### 10.5.3 C ABI Function Signatures

```c
// Mode 1: Streamed Two-Pass Sweep (O(M^2 * N_obs) RAM)
int qkrylov_ftlm_sweep_streamed_fp32(qkrylov_hamiltonian_h H, const float* beta_grid, int num_betas,
                                     const qkrylov_hamiltonian_h* observables, int num_observables,
                                     int n_random, int n_steps, uint64_t seed,
                                     qkrylov_ftlm_sweep_result_fp32_t* result);
int qkrylov_ftlm_sweep_streamed_fp64(qkrylov_hamiltonian_h H, const double* beta_grid, int num_betas,
                                     const qkrylov_hamiltonian_h* observables, int num_observables,
                                     int n_random, int n_steps, uint64_t seed,
                                     qkrylov_ftlm_sweep_result_fp64_t* result);

// Mode 2: Cached Stage 1 (Krylov Subspace Sampling)
int qkrylov_ftlm_sample_fp32(qkrylov_hamiltonian_h H, const qkrylov_hamiltonian_h* observables, int num_observables,
                             int n_random, int n_steps, uint64_t seed, qkrylov_ftlm_samples_h* out_samples);
int qkrylov_ftlm_sample_fp64(qkrylov_hamiltonian_h H, const qkrylov_hamiltonian_h* observables, int num_observables,
                             int n_random, int n_steps, uint64_t seed, qkrylov_ftlm_samples_h* out_samples);

// Mode 2: Cached Stage 2 (Fast Zero-SpMV Evaluation on Arbitrary Betas)
int qkrylov_ftlm_evaluate_sweep_fp32(qkrylov_ftlm_samples_h samples, const float* beta_grid, int num_betas,
                                     qkrylov_ftlm_sweep_result_fp32_t* result);
int qkrylov_ftlm_evaluate_sweep_fp64(qkrylov_ftlm_samples_h samples, const double* beta_grid, int num_betas,
                                     qkrylov_ftlm_sweep_result_fp64_t* result);

// Default combined sweep (aliases cached workflow)
int qkrylov_ftlm_sweep_fp32(qkrylov_hamiltonian_h H, const float* beta_grid, int num_betas,
                            const qkrylov_hamiltonian_h* observables, int num_observables,
                            int n_random, int n_steps, uint64_t seed,
                            qkrylov_ftlm_sweep_result_fp32_t* result);
int qkrylov_ftlm_sweep_fp64(qkrylov_hamiltonian_h H, const double* beta_grid, int num_betas,
                            const qkrylov_hamiltonian_h* observables, int num_observables,
                            int n_random, int n_steps, uint64_t seed,
                            qkrylov_ftlm_sweep_result_fp64_t* result);

// Single temperature point evaluation
int qkrylov_ftlm_fp32(qkrylov_hamiltonian_h H, float beta, int n_random, int n_steps, qkrylov_ftlm_result_fp32_t* result);
int qkrylov_ftlm_fp64(qkrylov_hamiltonian_h H, double beta, int n_random, int n_steps, qkrylov_ftlm_result_fp64_t* result);

// Memory cleanup for sweep results & sample handles
void qkrylov_ftlm_sweep_result_free_fp32(qkrylov_ftlm_sweep_result_fp32_t* result);
void qkrylov_ftlm_sweep_result_free_fp64(qkrylov_ftlm_sweep_result_fp64_t* result);
void qkrylov_ftlm_samples_destroy(qkrylov_ftlm_samples_h samples);
int  qkrylov_ftlm_samples_precision(qkrylov_ftlm_samples_h samples); // 0: FP32, 1: FP64
```

---

### 10.6 Pure-State Real-Time Dynamics (`time_evolve`)

Propagates an arbitrary initial pure state $|\psi(0)\rangle$ under unitary time evolution:
$$|\psi(t)\rangle = e^{-i \hat{H} t} |\psi(0)\rangle$$
and evaluates time-dependent survival probabilities $\mathcal{L}(t) = |\langle \psi(0) | \psi(t) \rangle|^2$ and complex expectation values $\langle \hat{O} \rangle(t) = \langle \psi(t) | \hat{O} | \psi(t) \rangle$.

```c
typedef struct {
    int num_times;
    int num_observables;
    const float* time_grid;
    const float* survival_probabilities_re;
    const float* survival_probabilities_im;
    const float* observable_expectations_re; /* Row-major: num_times x num_observables */
    const float* observable_expectations_im; /* Row-major: num_times x num_observables */
} qkrylov_real_time_result_fp32_t;

typedef struct {
    int num_times;
    int num_observables;
    const double* time_grid;
    const double* survival_probabilities_re;
    const double* survival_probabilities_im;
    const double* observable_expectations_re; /* Row-major: num_times x num_observables */
    const double* observable_expectations_im; /* Row-major: num_times x num_observables */
} qkrylov_real_time_result_fp64_t;

int qkrylov_time_evolve_fp32(
    qkrylov_hamiltonian_h H,
    const float* psi0_complex,
    const float* time_grid, int num_times,
    const qkrylov_hamiltonian_h* observables, int num_observables,
    int n_steps,
    qkrylov_real_time_result_fp32_t* result
);
int qkrylov_time_evolve_fp64(
    qkrylov_hamiltonian_h H,
    const double* psi0_complex,
    const double* time_grid, int num_times,
    const qkrylov_hamiltonian_h* observables, int num_observables,
    int n_steps,
    qkrylov_real_time_result_fp64_t* result
);

void qkrylov_real_time_result_free_fp32(qkrylov_real_time_result_fp32_t* result);
void qkrylov_real_time_result_free_fp64(qkrylov_real_time_result_fp64_t* result);
```

---

### 10.7 Finite-Temperature Dynamical Correlators (`ftlm_dynamics`)

Computes the unequal-time thermal correlation function:
$$C_{AB}(t; \beta) = \langle \hat{A}(t) \hat{B}(0) \rangle_\beta = \frac{1}{Z(\beta)} \operatorname{Tr}\left( e^{-\beta \hat{H}} e^{i \hat{H} t} \hat{A} e^{-i \hat{H} t} \hat{B} \right)$$
using **dual-Krylov propagation**: forms $|\chi_0\rangle = \hat{B} |\phi\rangle$ in the full Hilbert space and evolves it in its own Krylov subspace, eliminating projected operator truncation error and yielding machine-precision accuracy at $t=0$.

```c
typedef struct {
    float beta;
    int num_times;
    const float* time_grid;
    const float* correlations_re;
    const float* correlations_im;
    const float* correlation_errors;
} qkrylov_ftlm_dynamics_result_fp32_t;

typedef struct {
    double beta;
    int num_times;
    const double* time_grid;
    const double* correlations_re;
    const double* correlations_im;
    const double* correlation_errors;
} qkrylov_ftlm_dynamics_result_fp64_t;

int qkrylov_ftlm_dynamics_fp32(
    qkrylov_hamiltonian_h H,
    float beta,
    qkrylov_hamiltonian_h A,
    qkrylov_hamiltonian_h B,
    const float* time_grid, int num_times,
    int n_random, int n_steps, uint64_t seed,
    qkrylov_ftlm_dynamics_result_fp32_t* result
);
int qkrylov_ftlm_dynamics_fp64(
    qkrylov_hamiltonian_h H,
    double beta,
    qkrylov_hamiltonian_h A,
    qkrylov_hamiltonian_h B,
    const double* time_grid, int num_times,
    int n_random, int n_steps, uint64_t seed,
    qkrylov_ftlm_dynamics_result_fp64_t* result
);

void qkrylov_ftlm_dynamics_result_free_fp32(qkrylov_ftlm_dynamics_result_fp32_t* result);
void qkrylov_ftlm_dynamics_result_free_fp64(qkrylov_ftlm_dynamics_result_fp64_t* result);
```

---

### 10.8 Correction Vector Method

```c
typedef struct { double spectral_function; int iterations; int converged; } qkrylov_correction_vector_result_fp64_t;

int qkrylov_solver_correction_vector_fp64(
    qkrylov_hamiltonian_h h,
    const double* op_psi0_complex, double e0, double omega, double eta,
    int max_iter, double tol,
    qkrylov_correction_vector_result_fp64_t* result,
    double* correction_vector_out_complex
);
```

---

## 11. Parallel Vector BLAS-1 API (Host & Device-Resident)

Declared in [`include/qkrylov/c_api.h`](file:///home/pritipriya/Documents/GitHub/qkrylov/include/qkrylov/c_api.h#L496-L634):

### 11.1 Host Pointer BLAS-1 (Internally Dispatched to Kokkos)
```c
int qkrylov_vector_dot_fp64(uint64_t dim, const double* x, const double* y, double* dot_re, double* dot_im);
int qkrylov_vector_norm_fp64(uint64_t dim, const double* x, double* norm_out);
int qkrylov_vector_axpy_fp64(uint64_t dim, double a_re, double a_im, const double* x, double* y);
int qkrylov_vector_scal_fp64(uint64_t dim, double a_re, double a_im, double* x);
int qkrylov_vector_normalize_fp64(uint64_t dim, double* x);
int qkrylov_vector_zero_fill_fp64(uint64_t dim, double* x);
int qkrylov_vector_copy_fp64(uint64_t dim, const double* src, double* dst);
```

### 11.2 Device-Resident Vector API (Zero-Copy GPU Memory)
```c
// Lifecycle
qkrylov_device_vector_h qkrylov_device_vector_create_fp64(uint64_t dim);
void                    qkrylov_device_vector_destroy(qkrylov_device_vector_h vec);
uint64_t                qkrylov_device_vector_dimension(qkrylov_device_vector_h vec);
int                     qkrylov_device_vector_precision(qkrylov_device_vector_h vec);
void*                   qkrylov_device_vector_data_fp64(qkrylov_device_vector_h vec);

// Staging Transfers
int qkrylov_device_vector_copy_from_host_fp64(qkrylov_device_vector_h dst, const double* host_src_complex);
int qkrylov_device_vector_copy_to_host_fp64(const qkrylov_device_vector_h src, double* host_dst_complex);

// Parallel Device Kernels
int qkrylov_device_vector_dot_fp64(const qkrylov_device_vector_h x, const qkrylov_device_vector_h y, double* dot_re, double* dot_im);
int qkrylov_device_vector_norm_fp64(const qkrylov_device_vector_h x, double* norm_out);
int qkrylov_device_vector_axpy_fp64(double a_re, double a_im, const qkrylov_device_vector_h x, qkrylov_device_vector_h y);
int qkrylov_device_vector_scal_fp64(double a_re, double a_im, qkrylov_device_vector_h x);
int qkrylov_device_vector_normalize_fp64(qkrylov_device_vector_h x);
int qkrylov_device_vector_zero_fill_fp64(qkrylov_device_vector_h x);
int qkrylov_device_vector_copy_fp64(const qkrylov_device_vector_h src, qkrylov_device_vector_h dst);
```

---

## 12. Complete C Program Usage Example

A self-contained C application demonstrating full lifecycle management, error handling, Hamiltonian construction, and two-pass Lanczos ground state calculation:

```c
#include <qkrylov/c_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    printf("=== qkrylov C ABI Demo ===\n");

    // 1. Create Sz = 0 symmetry sector
    qkrylov_sector_h sector = qkrylov_sector_create();
    if (!sector) {
        fprintf(stderr, "Failed to create sector: %s\n", qkrylov_get_last_error_message());
        return 1;
    }
    qkrylov_sector_set_sz(sector, 0); // 2 * Sz = 0

    // 2. Create 4-site Spin-1/2 Basis and Site
    int N = 4;
    qkrylov_basis_h basis = qkrylov_spinhalf_basis_create(N, sector);
    qkrylov_site_h site = qkrylov_spinhalf_site_create();

    uint64_t dim = qkrylov_basis_dimension(basis);
    printf("Hilbert space dimension (N=%d, Sz=0): %llu\n", N, (unsigned long long)dim);

    // 3. Build Heisenberg model terms in OpSum: H = \sum_i [ Sz_i Sz_{i+1} + 0.5(Sp_i Sm_{i+1} + Sm_i Sp_{i+1}) ]
    qkrylov_opsum_h ops = qkrylov_opsum_create();
    for (int i = 0; i < N - 1; ++i) {
        qkrylov_opsum_add_term_2body_fp64(ops, 1.0, 0.0, "Sz", i, "Sz", i + 1);
        qkrylov_opsum_add_term_2body_fp64(ops, 0.5, 0.0, "Sp", i, "Sm", i + 1);
        qkrylov_opsum_add_term_2body_fp64(ops, 0.5, 0.0, "Sm", i, "Sp", i + 1);
    }

    // 4. Construct Matrix-Free Hamiltonian
    qkrylov_hamiltonian_h H = qkrylov_hamiltonian_create_fp64(basis, site, ops);
    if (!H) {
        fprintf(stderr, "Failed to build Hamiltonian: %s\n", qkrylov_get_last_error_message());
        return 1;
    }

    // 5. Solve for ground state using Two-Pass Lanczos (strictly O(N) memory)
    qkrylov_lanczos_result_fp64_t res;
    double* psi0_complex = (double*)malloc(2 * dim * sizeof(double));

    int status = qkrylov_lanczos_two_pass_ground_state_complex_fp64(H, 100, 1e-12, &res, psi0_complex);
    if (status != QKRYLOV_SUCCESS) {
        fprintf(stderr, "Lanczos solver failed: %s\n", qkrylov_get_last_error_message());
        free(psi0_complex);
        return 1;
    }

    printf("Ground state energy E0 = %.10f\n", res.energy);
    printf("Iterations = %d, Converged = %s\n", res.iterations, res.converged ? "YES" : "NO");

    // 6. Verify norm using BLAS-1 kernel
    double norm_val = 0.0;
    qkrylov_vector_norm_fp64(dim, psi0_complex, &norm_val);
    printf("Computed eigenvector norm = %.10f\n", norm_val);

    // 7. Decoupled FTLM (Stage 1 Sampling + Stage 2 Multi-Temperature Evaluation)
    qkrylov_ftlm_samples_h samples = NULL;
    qkrylov_hamiltonian_h observables[1] = { H };
    int sample_status = qkrylov_ftlm_sample_fp64(H, observables, 1, 20, 25, 42, &samples);
    if (sample_status == QKRYLOV_SUCCESS) {
        double betas[3] = { 0.2, 1.0, 5.0 };
        qkrylov_ftlm_sweep_result_fp64_t sweep_res;
        if (qkrylov_ftlm_evaluate_sweep_fp64(samples, betas, 3, &sweep_res) == QKRYLOV_SUCCESS) {
            printf("FTLM Sweep: beta=%.1f -> E=%.6f, Cv=%.6f\n", sweep_res.beta_grid[1], sweep_res.internal_energies[1], sweep_res.specific_heats[1]);
            qkrylov_ftlm_sweep_result_free_fp64(&sweep_res);
        }
        qkrylov_ftlm_samples_destroy(samples);
    }

    // 8. Cleanup opaque handles and host memory
    free(psi0_complex);
    qkrylov_hamiltonian_destroy(H);
    qkrylov_opsum_destroy(ops);
    qkrylov_site_destroy(site);
    qkrylov_basis_destroy(basis);
    qkrylov_sector_destroy(sector);

    printf("=== Completed successfully ===\n");
    return 0;
}
```

### Compiling and Linking the C Example
```bash
gcc -std=c11 -O3 main.c -I/path/to/qkrylov/include -L/path/to/qkrylov/build -lqkrylov -o demo
LD_LIBRARY_PATH=/path/to/qkrylov/build ./demo
```
