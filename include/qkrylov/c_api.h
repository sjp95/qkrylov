#ifndef QKRYLOV_C_API_H
#define QKRYLOV_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef QKRYLOV_API
#if defined(_WIN32) || defined(__CYGWIN__)
#define QKRYLOV_API
#else
#define QKRYLOV_API __attribute__((visibility("default")))
#endif
#endif

/* -----------------------------------------------------------------------------
 * Return Error Codes & Precision IDs
 * ----------------------------------------------------------------------------- */
#define QKRYLOV_SUCCESS                0
#define QKRYLOV_ERROR_INVALID_ARG     -1
#define QKRYLOV_ERROR_EXCEPTION       -2

#define QKRYLOV_PRECISION_FP32         0
#define QKRYLOV_PRECISION_FP64         1

/* -----------------------------------------------------------------------------
 * Error Diagnostics API
 * ----------------------------------------------------------------------------- */
/* Returns the last error message caught across the C ABI barrier on the current calling thread.
 * Returns an empty string "" if no error occurred. The returned pointer is valid until the next
 * C ABI call or until qkrylov_clear_last_error() is called on the current thread. */
QKRYLOV_API const char* qkrylov_get_last_error_message(void);

/* Clears the thread-local error message for the current calling thread. */
QKRYLOV_API void        qkrylov_clear_last_error(void);

/* -----------------------------------------------------------------------------
 * Opaque Handles
 * ----------------------------------------------------------------------------- */
typedef struct qkrylov_sector_t       qkrylov_sector_t;
typedef struct qkrylov_sector_t*      qkrylov_sector_h;
typedef struct qkrylov_basis_t        qkrylov_basis_t;
typedef struct qkrylov_basis_t*       qkrylov_basis_h;
typedef struct qkrylov_site_t         qkrylov_site_t;
typedef struct qkrylov_site_t*        qkrylov_site_h;
typedef struct qkrylov_opsum_t        qkrylov_opsum_t;
typedef struct qkrylov_opsum_t*       qkrylov_opsum_h;
typedef struct qkrylov_hamiltonian_t  qkrylov_hamiltonian_t;
typedef struct qkrylov_hamiltonian_t* qkrylov_hamiltonian_h;
typedef struct qkrylov_device_vector_t  qkrylov_device_vector_t;
typedef struct qkrylov_device_vector_t* qkrylov_device_vector_h;
typedef qkrylov_device_vector_h         qkrylov_device_vector_fp64_h;
typedef qkrylov_device_vector_h         qkrylov_device_vector_fp32_h;

/* -----------------------------------------------------------------------------
 * Sector API
 * ----------------------------------------------------------------------------- */
QKRYLOV_API qkrylov_sector_h qkrylov_sector_create(void);
QKRYLOV_API void             qkrylov_sector_destroy(qkrylov_sector_h sector);
QKRYLOV_API int              qkrylov_sector_set_sz(qkrylov_sector_h sector, int sz2);
QKRYLOV_API int              qkrylov_sector_set_hubbard_particles(qkrylov_sector_h sector, int nup, int ndn);
QKRYLOV_API int              qkrylov_sector_set_n(qkrylov_sector_h sector, int n);
QKRYLOV_API int              qkrylov_sector_set_nb(qkrylov_sector_h sector, int nb);
QKRYLOV_API int              qkrylov_sector_get_sz(qkrylov_sector_h sector, int* sz2_out, int* active_out);
QKRYLOV_API int              qkrylov_sector_get_hubbard_particles(qkrylov_sector_h sector, int* nup_out, int* ndn_out, int* active_out);
QKRYLOV_API int              qkrylov_sector_get_n(qkrylov_sector_h sector, int* n_out, int* active_out);
QKRYLOV_API int              qkrylov_sector_get_nb(qkrylov_sector_h sector, int* nb_out, int* active_out);

/* -----------------------------------------------------------------------------
 * Basis API
 * ----------------------------------------------------------------------------- */
QKRYLOV_API qkrylov_basis_h  qkrylov_spinhalf_basis_create(int num_sites, qkrylov_sector_h sector);
QKRYLOV_API qkrylov_basis_h  qkrylov_basis_create_spin_s(int N, double S, const qkrylov_sector_t* sector);
QKRYLOV_API qkrylov_basis_h  qkrylov_fermion_basis_create(int num_sites, qkrylov_sector_h sector);
QKRYLOV_API qkrylov_basis_h  qkrylov_hubbard_basis_create(int num_sites, qkrylov_sector_h sector);
QKRYLOV_API qkrylov_basis_h  qkrylov_tj_basis_create(int num_sites, qkrylov_sector_h sector);
QKRYLOV_API void             qkrylov_basis_destroy(qkrylov_basis_h basis);
QKRYLOV_API uint64_t         qkrylov_basis_dimension(qkrylov_basis_h basis);
QKRYLOV_API int              qkrylov_basis_nsites(qkrylov_basis_h basis);
QKRYLOV_API uint64_t         qkrylov_basis_state(qkrylov_basis_h basis, uint64_t index);
QKRYLOV_API int64_t          qkrylov_basis_index(qkrylov_basis_h basis, uint64_t state_bitstring);
QKRYLOV_API int              qkrylov_basis_contains(qkrylov_basis_h basis, uint64_t state_bitstring);

/* Basis Reflection */
typedef enum {
    QKRYLOV_BASIS_SPIN_HALF = 0,
    QKRYLOV_BASIS_SPIN_S    = 1,
    QKRYLOV_BASIS_FERMION   = 2,
    QKRYLOV_BASIS_HUBBARD   = 3,
    QKRYLOV_BASIS_TJ        = 4
} qkrylov_basis_type_t;

QKRYLOV_API int              qkrylov_basis_get_type(qkrylov_basis_h basis, qkrylov_basis_type_t* type_out);
QKRYLOV_API int              qkrylov_basis_get_spin(qkrylov_basis_h basis, double* spin_out);
QKRYLOV_API int              qkrylov_basis_get_dimension_per_site(qkrylov_basis_h basis, int* d_out);
QKRYLOV_API qkrylov_sector_h  qkrylov_basis_get_sector(qkrylov_basis_h basis);

/* -----------------------------------------------------------------------------
 * Site API
 * ----------------------------------------------------------------------------- */
QKRYLOV_API qkrylov_site_h   qkrylov_spinhalf_site_create(void);
QKRYLOV_API qkrylov_site_h   qkrylov_site_create_spin_s(double S);
QKRYLOV_API qkrylov_site_h   qkrylov_fermion_site_create(void);
QKRYLOV_API qkrylov_site_h   qkrylov_hubbard_site_create(void);
QKRYLOV_API qkrylov_site_h   qkrylov_tj_site_create(void);
QKRYLOV_API void             qkrylov_site_destroy(qkrylov_site_h site);

/* Site Reflection & Local Action Evaluation */
typedef enum {
    QKRYLOV_SITE_SPIN_HALF = 0,
    QKRYLOV_SITE_SPIN_S    = 1,
    QKRYLOV_SITE_FERMION   = 2,
    QKRYLOV_SITE_HUBBARD   = 3,
    QKRYLOV_SITE_TJ        = 4
} qkrylov_site_type_t;

typedef struct {
    int      valid;             /* 1 if transition non-zero, 0 if annihilated */
    uint64_t new_state;         /* Updated Fock state bitstring */
    double   matrix_element_re; /* Real part of transition matrix element */
    double   matrix_element_im; /* Imaginary part of transition matrix element */
} qkrylov_local_action_t;

QKRYLOV_API int              qkrylov_site_apply(qkrylov_site_h site, const char* op, int site_idx,
                                                uint64_t state, qkrylov_local_action_t* action_out);
QKRYLOV_API int              qkrylov_site_get_type(qkrylov_site_h site, qkrylov_site_type_t* type_out);
QKRYLOV_API int              qkrylov_site_get_spin(qkrylov_site_h site, double* spin_out);
QKRYLOV_API int              qkrylov_site_get_dimension_per_site(qkrylov_site_h site, int* d_out);

/* -----------------------------------------------------------------------------
 * OpSum API
 * ----------------------------------------------------------------------------- */
QKRYLOV_API qkrylov_opsum_h  qkrylov_opsum_create(void);
QKRYLOV_API void             qkrylov_opsum_destroy(qkrylov_opsum_h opsum);
QKRYLOV_API int              qkrylov_opsum_clear(qkrylov_opsum_h opsum);

/* OpSum Reflection */
QKRYLOV_API int              qkrylov_opsum_size(qkrylov_opsum_h opsum, int* size_out);
QKRYLOV_API int              qkrylov_opsum_get_term_info(qkrylov_opsum_h opsum, int term_idx,
                                                         double* coeff_re_out, double* coeff_im_out,
                                                         int* num_factors_out);
QKRYLOV_API int              qkrylov_opsum_get_factor(qkrylov_opsum_h opsum, int term_idx, int factor_idx,
                                                      char* op_buf, int op_buf_len, int* site_out);

/* Single-Precision (FP32) OpSum Add Terms */
QKRYLOV_API int              qkrylov_opsum_add_term_1body_fp32(qkrylov_opsum_h opsum, float coeff_real, float coeff_imag,
                                                               const char* op1, int site1);
QKRYLOV_API int              qkrylov_opsum_add_term_2body_fp32(qkrylov_opsum_h opsum, float coeff_real, float coeff_imag,
                                                               const char* op1, int site1,
                                                               const char* op2, int site2);
QKRYLOV_API int              qkrylov_opsum_add_term_nbody_fp32(qkrylov_opsum_h opsum, float coeff_real, float coeff_imag,
                                                               int n_factors, const char** ops, const int* sites);

/* Double-Precision (FP64) OpSum Add Terms */
QKRYLOV_API int              qkrylov_opsum_add_term_1body_fp64(qkrylov_opsum_h opsum, double coeff_real, double coeff_imag,
                                                               const char* op1, int site1);
QKRYLOV_API int              qkrylov_opsum_add_term_2body_fp64(qkrylov_opsum_h opsum, double coeff_real, double coeff_imag,
                                                               const char* op1, int site1,
                                                               const char* op2, int site2);
QKRYLOV_API int              qkrylov_opsum_add_term_nbody_fp64(qkrylov_opsum_h opsum, double coeff_real, double coeff_imag,
                                                               int n_factors, const char** ops, const int* sites);

/* Default (FP64) Convenience Aliases */
QKRYLOV_API int              qkrylov_opsum_add_term_1body(qkrylov_opsum_h opsum, double coeff_real, double coeff_imag,
                                                          const char* op1, int site1);
QKRYLOV_API int              qkrylov_opsum_add_term_2body(qkrylov_opsum_h opsum, double coeff_real, double coeff_imag,
                                                          const char* op1, int site1,
                                                          const char* op2, int site2);
QKRYLOV_API int              qkrylov_opsum_add_term_nbody(qkrylov_opsum_h opsum, double coeff_real, double coeff_imag,
                                                          int n_factors, const char** ops, const int* sites);

/* -----------------------------------------------------------------------------
 * Device & Hardware Query API
 * ----------------------------------------------------------------------------- */
QKRYLOV_API int         qkrylov_is_gpu_build(void);
QKRYLOV_API const char* qkrylov_find_gpu(void);
QKRYLOV_API int         qkrylov_gpu_count(void);
QKRYLOV_API int         qkrylov_initialize_device(const char* device_str);

/* -----------------------------------------------------------------------------
 * Matrix-Free Hamiltonian API
 * ----------------------------------------------------------------------------- */
QKRYLOV_API void                  qkrylov_hamiltonian_destroy(qkrylov_hamiltonian_h h);
QKRYLOV_API uint64_t              qkrylov_hamiltonian_dimension(qkrylov_hamiltonian_h h);
QKRYLOV_API int                   qkrylov_hamiltonian_precision(qkrylov_hamiltonian_h h);

/* Single-Precision (FP32) Hamiltonian API */
QKRYLOV_API qkrylov_hamiltonian_h qkrylov_hamiltonian_create_fp32(qkrylov_basis_h basis,
                                                                 qkrylov_site_h site,
                                                                 qkrylov_opsum_h opsum);
QKRYLOV_API qkrylov_hamiltonian_h qkrylov_hamiltonian_create_device_fp32(qkrylov_basis_h basis,
                                                                        qkrylov_site_h site,
                                                                        qkrylov_opsum_h opsum,
                                                                        const char* device_str);
QKRYLOV_API int                   qkrylov_hamiltonian_apply_fp32(qkrylov_hamiltonian_h h,
                                                                 const float* x_real, const float* x_imag,
                                                                 float* y_real, float* y_imag);
QKRYLOV_API int                   qkrylov_hamiltonian_apply_complex_fp32(qkrylov_hamiltonian_h h,
                                                                         const float* x_complex,
                                                                         float* y_complex);
QKRYLOV_API int                   qkrylov_hamiltonian_apply_device_fp32(qkrylov_hamiltonian_h h,
                                                                        const qkrylov_device_vector_h x_dev,
                                                                        qkrylov_device_vector_h y_dev);
QKRYLOV_API int                   qkrylov_hamiltonian_diagonal_fp32(qkrylov_hamiltonian_h h,
                                                                    float* diag_out);
QKRYLOV_API int                   qkrylov_hamiltonian_diagonal_device_fp32(qkrylov_hamiltonian_h h,
                                                                           qkrylov_device_vector_h diag_out);

/* Double-Precision (FP64) Hamiltonian API */
QKRYLOV_API qkrylov_hamiltonian_h qkrylov_hamiltonian_create_fp64(qkrylov_basis_h basis,
                                                                 qkrylov_site_h site,
                                                                 qkrylov_opsum_h opsum);
QKRYLOV_API qkrylov_hamiltonian_h qkrylov_hamiltonian_create_device_fp64(qkrylov_basis_h basis,
                                                                        qkrylov_site_h site,
                                                                        qkrylov_opsum_h opsum,
                                                                        const char* device_str);
QKRYLOV_API int                   qkrylov_hamiltonian_apply_fp64(qkrylov_hamiltonian_h h,
                                                                 const double* x_real, const double* x_imag,
                                                                 double* y_real, double* y_imag);
QKRYLOV_API int                   qkrylov_hamiltonian_apply_complex_fp64(qkrylov_hamiltonian_h h,
                                                                         const double* x_complex,
                                                                         double* y_complex);
QKRYLOV_API int                   qkrylov_hamiltonian_apply_device_fp64(qkrylov_hamiltonian_h h,
                                                                        const qkrylov_device_vector_h x_dev,
                                                                        qkrylov_device_vector_h y_dev);
QKRYLOV_API int                   qkrylov_hamiltonian_diagonal_fp64(qkrylov_hamiltonian_h h,
                                                                    double* diag_out);
QKRYLOV_API int                   qkrylov_hamiltonian_diagonal_device_fp64(qkrylov_hamiltonian_h h,
                                                                           qkrylov_device_vector_h diag_out);

/* Default (FP64) Convenience Aliases */
QKRYLOV_API qkrylov_hamiltonian_h qkrylov_hamiltonian_create(qkrylov_basis_h basis,
                                                            qkrylov_site_h site,
                                                            qkrylov_opsum_h opsum);
QKRYLOV_API qkrylov_hamiltonian_h qkrylov_hamiltonian_create_device(qkrylov_basis_h basis,
                                                                    qkrylov_site_h site,
                                                                    qkrylov_opsum_h opsum,
                                                                    const char* device_str);
QKRYLOV_API int                   qkrylov_hamiltonian_apply(qkrylov_hamiltonian_h h,
                                                            const double* x_real, const double* x_imag,
                                                            double* y_real, double* y_imag);
QKRYLOV_API int                   qkrylov_hamiltonian_apply_complex(qkrylov_hamiltonian_h h,
                                                                    const double* x_complex,
                                                                    double* y_complex);
QKRYLOV_API int                   qkrylov_hamiltonian_apply_device(qkrylov_hamiltonian_h h,
                                                                   const qkrylov_device_vector_h x_dev,
                                                                   qkrylov_device_vector_h y_dev);
QKRYLOV_API int                   qkrylov_hamiltonian_diagonal(qkrylov_hamiltonian_h h,
                                                               double* diag_out);
QKRYLOV_API int                   qkrylov_hamiltonian_diagonal_device(qkrylov_hamiltonian_h h,
                                                                      qkrylov_device_vector_h diag_out);

/* -----------------------------------------------------------------------------
 * Solvers API & Result Structs
 * ----------------------------------------------------------------------------- */
typedef struct {
    float energy;
    int iterations;
    int converged;
} qkrylov_lanczos_result_fp32_t;

typedef struct {
    double energy;
    int iterations;
    int converged;
} qkrylov_lanczos_result_fp64_t;

typedef qkrylov_lanczos_result_fp64_t qkrylov_lanczos_result_c_t;

typedef struct {
    int iterations;
    int converged;
} qkrylov_davidson_result_c_t;

typedef struct {
    int iterations;
    int converged;
} qkrylov_lanczos_lowest_result_c_t;

typedef struct {
    float beta;
    float partition_function;
    float internal_energy;
    float specific_heat;
} qkrylov_ftlm_result_fp32_t;

typedef struct {
    double beta;
    double partition_function;
    double internal_energy;
    double specific_heat;
} qkrylov_ftlm_result_fp64_t;

typedef qkrylov_ftlm_result_fp64_t qkrylov_ftlm_result_c_t;

typedef struct {
    float spectral_function;
    int iterations;
    int converged;
} qkrylov_correction_vector_result_fp32_t;

typedef struct {
    double spectral_function;
    int iterations;
    int converged;
} qkrylov_correction_vector_result_fp64_t;

typedef qkrylov_correction_vector_result_fp64_t qkrylov_correction_vector_result_c_t;

/* Single-Precision (FP32) Solvers */
QKRYLOV_API int   qkrylov_lanczos_ground_state_fp32(qkrylov_hamiltonian_h h,
                                                    int maxiter,
                                                    float tol,
                                                    qkrylov_lanczos_result_fp32_t* result);
QKRYLOV_API int   qkrylov_lanczos_ground_state_complex_fp32(qkrylov_hamiltonian_h h,
                                                            int maxiter,
                                                            float tol,
                                                            qkrylov_lanczos_result_fp32_t* result,
                                                            float* eigenvector_complex);
QKRYLOV_API int   qkrylov_lanczos_two_pass_ground_state_fp32(qkrylov_hamiltonian_h h,
                                                             int maxiter,
                                                             float tol,
                                                             qkrylov_lanczos_result_fp32_t* result);
QKRYLOV_API int   qkrylov_lanczos_two_pass_ground_state_complex_fp32(qkrylov_hamiltonian_h h,
                                                                     int maxiter,
                                                                     float tol,
                                                                     qkrylov_lanczos_result_fp32_t* result,
                                                                     float* eigenvector_complex);
QKRYLOV_API int   qkrylov_lanczos_lowest_complex_fp32(qkrylov_hamiltonian_h h,
                                                      int n_eig,
                                                      int maxiter,
                                                      float tol,
                                                      float* eigenvalues_out,
                                                      float* eigenvectors_complex_out,
                                                      qkrylov_lanczos_lowest_result_c_t* result_info,
                                                      const float* initial_vector_complex);
QKRYLOV_API int   qkrylov_davidson_lowest_complex_fp32(qkrylov_hamiltonian_h h,
                                                       int n_eig,
                                                       int max_subspace,
                                                       float tol,
                                                       float* eigenvalues_out,
                                                       float* eigenvectors_complex_out,
                                                       qkrylov_davidson_result_c_t* result_info);
QKRYLOV_API int   qkrylov_continued_fraction_coeffs_complex_fp32(qkrylov_hamiltonian_h h,
                                                                 const float* phi0_complex,
                                                                 int n_iter,
                                                                 float* alphas_out,
                                                                 float* betas_out,
                                                                 float* norm_phi0_out,
                                                                 int* num_coeffs_out);
QKRYLOV_API float qkrylov_evaluate_spectral_function_fp32(const float* alphas,
                                                          const float* betas,
                                                          size_t n,
                                                          float norm_phi0,
                                                          float omega,
                                                          float E0,
                                                          float eta);
QKRYLOV_API int   qkrylov_ftlm_fp32(qkrylov_hamiltonian_h h,
                                    float beta,
                                    int n_random,
                                    int n_steps,
                                    qkrylov_ftlm_result_fp32_t* result);
QKRYLOV_API int   qkrylov_solver_correction_vector_fp32(qkrylov_hamiltonian_h h,
                                                        const float* op_psi0_complex,
                                                        float e0,
                                                        float omega,
                                                        float eta,
                                                        int max_iter,
                                                        float tol,
                                                        qkrylov_correction_vector_result_fp32_t* result,
                                                        float* correction_vector_out_complex);

/* Single-Precision (FP32) Vector Operations */
QKRYLOV_API int   qkrylov_vector_dot_fp32(uint64_t dim, const float* x_complex, const float* y_complex, float* dot_re, float* dot_im);
QKRYLOV_API int   qkrylov_vector_norm_fp32(uint64_t dim, const float* x_complex, float* norm_out);
QKRYLOV_API int   qkrylov_vector_axpy_fp32(uint64_t dim, float a_re, float a_im, const float* x_complex, float* y_complex);
QKRYLOV_API int   qkrylov_vector_scal_fp32(uint64_t dim, float a_re, float a_im, float* x_complex);
QKRYLOV_API int   qkrylov_vector_normalize_fp32(uint64_t dim, float* x_complex);
QKRYLOV_API int   qkrylov_vector_zero_fill_fp32(uint64_t dim, float* x_complex);
QKRYLOV_API int   qkrylov_vector_copy_fp32(uint64_t dim, const float* src_complex, float* dst_complex);

/* Double-Precision (FP64) Solvers */
QKRYLOV_API int    qkrylov_lanczos_ground_state_fp64(qkrylov_hamiltonian_h h,
                                                     int maxiter,
                                                     double tol,
                                                     qkrylov_lanczos_result_fp64_t* result);
QKRYLOV_API int    qkrylov_lanczos_ground_state_complex_fp64(qkrylov_hamiltonian_h h,
                                                              int maxiter,
                                                              double tol,
                                                              qkrylov_lanczos_result_fp64_t* result,
                                                              double* eigenvector_complex);
QKRYLOV_API int    qkrylov_lanczos_two_pass_ground_state_fp64(qkrylov_hamiltonian_h h,
                                                              int maxiter,
                                                              double tol,
                                                              qkrylov_lanczos_result_fp64_t* result);
QKRYLOV_API int    qkrylov_lanczos_two_pass_ground_state_complex_fp64(qkrylov_hamiltonian_h h,
                                                                      int maxiter,
                                                                      double tol,
                                                                      qkrylov_lanczos_result_fp64_t* result,
                                                                      double* eigenvector_complex);
QKRYLOV_API int    qkrylov_lanczos_lowest_complex_fp64(qkrylov_hamiltonian_h h,
                                                        int n_eig,
                                                        int maxiter,
                                                        double tol,
                                                        double* eigenvalues_out,
                                                        double* eigenvectors_complex_out,
                                                        qkrylov_lanczos_lowest_result_c_t* result_info,
                                                        const double* initial_vector_complex);
QKRYLOV_API int    qkrylov_davidson_lowest_complex_fp64(qkrylov_hamiltonian_h h,
                                                        int n_eig,
                                                       int max_subspace,
                                                       double tol,
                                                       double* eigenvalues_out,
                                                       double* eigenvectors_complex_out,
                                                       qkrylov_davidson_result_c_t* result_info);
QKRYLOV_API int    qkrylov_continued_fraction_coeffs_complex_fp64(qkrylov_hamiltonian_h h,
                                                                  const double* phi0_complex,
                                                                  int n_iter,
                                                                  double* alphas_out,
                                                                  double* betas_out,
                                                                  double* norm_phi0_out,
                                                                  int* num_coeffs_out);
QKRYLOV_API double qkrylov_evaluate_spectral_function_fp64(const double* alphas,
                                                           const double* betas,
                                                           size_t n,
                                                           double norm_phi0,
                                                           double omega,
                                                           double E0,
                                                           double eta);
QKRYLOV_API int    qkrylov_ftlm_fp64(qkrylov_hamiltonian_h h,
                                     double beta,
                                     int n_random,
                                     int n_steps,
                                     qkrylov_ftlm_result_fp64_t* result);
QKRYLOV_API int    qkrylov_solver_correction_vector_fp64(qkrylov_hamiltonian_h h,
                                                         const double* op_psi0_complex,
                                                         double e0,
                                                         double omega,
                                                         double eta,
                                                         int max_iter,
                                                         double tol,
                                                         qkrylov_correction_vector_result_fp64_t* result,
                                                         double* correction_vector_out_complex);

/* Double-Precision (FP64) Vector Operations */
QKRYLOV_API int    qkrylov_vector_dot_fp64(uint64_t dim, const double* x_complex, const double* y_complex, double* dot_re, double* dot_im);
QKRYLOV_API int    qkrylov_vector_norm_fp64(uint64_t dim, const double* x_complex, double* norm_out);
QKRYLOV_API int    qkrylov_vector_axpy_fp64(uint64_t dim, double a_re, double a_im, const double* x_complex, double* y_complex);
QKRYLOV_API int    qkrylov_vector_scal_fp64(uint64_t dim, double a_re, double a_im, double* x_complex);
QKRYLOV_API int    qkrylov_vector_normalize_fp64(uint64_t dim, double* x_complex);
QKRYLOV_API int    qkrylov_vector_zero_fill_fp64(uint64_t dim, double* x_complex);
QKRYLOV_API int    qkrylov_vector_copy_fp64(uint64_t dim, const double* src_complex, double* dst_complex);

/* Default (FP64) Convenience Aliases */
QKRYLOV_API int    qkrylov_lanczos_ground_state(qkrylov_hamiltonian_h h,
                                                int maxiter,
                                                double tol,
                                                qkrylov_lanczos_result_c_t* result);
QKRYLOV_API int    qkrylov_lanczos_ground_state_complex(qkrylov_hamiltonian_h h,
                                                        int maxiter,
                                                        double tol,
                                                        qkrylov_lanczos_result_c_t* result,
                                                        double* eigenvector_complex);
QKRYLOV_API int    qkrylov_lanczos_two_pass_ground_state(qkrylov_hamiltonian_h h,
                                                         int maxiter,
                                                         double tol,
                                                         qkrylov_lanczos_result_c_t* result);
QKRYLOV_API int    qkrylov_lanczos_two_pass_ground_state_complex(qkrylov_hamiltonian_h h,
                                                                 int maxiter,
                                                                 double tol,
                                                                 qkrylov_lanczos_result_c_t* result,
                                                                 double* eigenvector_complex);
QKRYLOV_API int    qkrylov_lanczos_lowest_complex(qkrylov_hamiltonian_h h,
                                                  int n_eig,
                                                  int maxiter,
                                                  double tol,
                                                  double* eigenvalues_out,
                                                  double* eigenvectors_complex_out,
                                                  qkrylov_lanczos_lowest_result_c_t* result_info,
                                                  const double* initial_vector_complex);
QKRYLOV_API int    qkrylov_davidson_lowest_complex(qkrylov_hamiltonian_h h,
                                                   int n_eig,
                                                   int max_subspace,
                                                   double tol,
                                                   double* eigenvalues_out,
                                                   double* eigenvectors_complex_out,
                                                   qkrylov_davidson_result_c_t* result_info);
QKRYLOV_API int    qkrylov_continued_fraction_coeffs_complex(qkrylov_hamiltonian_h h,
                                                             const double* phi0_complex,
                                                             int n_iter,
                                                             double* alphas_out,
                                                             double* betas_out,
                                                             double* norm_phi0_out,
                                                             int* num_coeffs_out);
QKRYLOV_API double qkrylov_evaluate_spectral_function(const double* alphas,
                                                      const double* betas,
                                                      size_t n,
                                                      double norm_phi0,
                                                      double omega,
                                                      double E0,
                                                      double eta);
QKRYLOV_API int    qkrylov_ftlm(qkrylov_hamiltonian_h h,
                                double beta,
                                int n_random,
                                int n_steps,
                                qkrylov_ftlm_result_c_t* result);
QKRYLOV_API int    qkrylov_solver_correction_vector(qkrylov_hamiltonian_h h,
                                                    const double* op_psi0_complex,
                                                    double e0,
                                                    double omega,
                                                    double eta,
                                                    int max_iter,
                                                    double tol,
                                                    qkrylov_correction_vector_result_c_t* result,
                                                    double* correction_vector_out_complex);

/* Default (FP64) Vector Operations */
QKRYLOV_API int    qkrylov_vector_dot(uint64_t dim, const double* x_complex, const double* y_complex, double* dot_re, double* dot_im);
QKRYLOV_API int    qkrylov_vector_norm(uint64_t dim, const double* x_complex, double* norm_out);
QKRYLOV_API int    qkrylov_vector_axpy(uint64_t dim, double a_re, double a_im, const double* x_complex, double* y_complex);
QKRYLOV_API int    qkrylov_vector_scal(uint64_t dim, double a_re, double a_im, double* x_complex);
QKRYLOV_API int    qkrylov_vector_normalize(uint64_t dim, double* x_complex);
QKRYLOV_API int    qkrylov_vector_zero_fill(uint64_t dim, double* x_complex);
QKRYLOV_API int    qkrylov_vector_copy(uint64_t dim, const double* src_complex, double* dst_complex);

/* =============================================================================
 * Device-Resident Vector API (Zero-Copy GPU Operations)
 * ============================================================================= */

/* Common Device Vector Management */
QKRYLOV_API void                    qkrylov_device_vector_destroy(qkrylov_device_vector_h vec);
QKRYLOV_API uint64_t                qkrylov_device_vector_dimension(qkrylov_device_vector_h vec);
QKRYLOV_API int                     qkrylov_device_vector_precision(qkrylov_device_vector_h vec);
QKRYLOV_API void*                   qkrylov_device_vector_data(qkrylov_device_vector_h vec);

/* Single-Precision (FP32) Device Vector Operations */
QKRYLOV_API qkrylov_device_vector_h qkrylov_device_vector_create_fp32(uint64_t dim);
QKRYLOV_API int                     qkrylov_device_vector_copy_from_host_fp32(qkrylov_device_vector_h dst, const float* host_src_complex);
QKRYLOV_API int                     qkrylov_device_vector_copy_to_host_fp32(const qkrylov_device_vector_h src, float* host_dst_complex);
QKRYLOV_API int                     qkrylov_device_vector_dot_fp32(const qkrylov_device_vector_h x, const qkrylov_device_vector_h y, float* dot_re, float* dot_im);
QKRYLOV_API int                     qkrylov_device_vector_norm_fp32(const qkrylov_device_vector_h x, float* norm_out);
QKRYLOV_API int                     qkrylov_device_vector_axpy_fp32(float a_re, float a_im, const qkrylov_device_vector_h x, qkrylov_device_vector_h y);
QKRYLOV_API int                     qkrylov_device_vector_scal_fp32(float a_re, float a_im, qkrylov_device_vector_h x);
QKRYLOV_API int                     qkrylov_device_vector_normalize_fp32(qkrylov_device_vector_h x);
QKRYLOV_API int                     qkrylov_device_vector_zero_fill_fp32(qkrylov_device_vector_h x);
QKRYLOV_API int                     qkrylov_device_vector_copy_fp32(const qkrylov_device_vector_h src, qkrylov_device_vector_h dst);
QKRYLOV_API void*                   qkrylov_device_vector_data_fp32(qkrylov_device_vector_h vec);

/* Double-Precision (FP64) Device Vector Operations */
QKRYLOV_API qkrylov_device_vector_h qkrylov_device_vector_create_fp64(uint64_t dim);
QKRYLOV_API int                     qkrylov_device_vector_copy_from_host_fp64(qkrylov_device_vector_h dst, const double* host_src_complex);
QKRYLOV_API int                     qkrylov_device_vector_copy_to_host_fp64(const qkrylov_device_vector_h src, double* host_dst_complex);
QKRYLOV_API int                     qkrylov_device_vector_dot_fp64(const qkrylov_device_vector_h x, const qkrylov_device_vector_h y, double* dot_re, double* dot_im);
QKRYLOV_API int                     qkrylov_device_vector_norm_fp64(const qkrylov_device_vector_h x, double* norm_out);
QKRYLOV_API int                     qkrylov_device_vector_axpy_fp64(double a_re, double a_im, const qkrylov_device_vector_h x, qkrylov_device_vector_h y);
QKRYLOV_API int                     qkrylov_device_vector_scal_fp64(double a_re, double a_im, qkrylov_device_vector_h x);
QKRYLOV_API int                     qkrylov_device_vector_normalize_fp64(qkrylov_device_vector_h x);
QKRYLOV_API int                     qkrylov_device_vector_zero_fill_fp64(qkrylov_device_vector_h x);
QKRYLOV_API int                     qkrylov_device_vector_copy_fp64(const qkrylov_device_vector_h src, qkrylov_device_vector_h dst);
QKRYLOV_API void*                   qkrylov_device_vector_data_fp64(qkrylov_device_vector_h vec);

/* Default (FP64) Device Vector Convenience Aliases */
QKRYLOV_API qkrylov_device_vector_h qkrylov_device_vector_create(uint64_t dim);
QKRYLOV_API int                     qkrylov_device_vector_copy_from_host(qkrylov_device_vector_h dst, const double* host_src_complex);
QKRYLOV_API int                     qkrylov_device_vector_copy_to_host(const qkrylov_device_vector_h src, double* host_dst_complex);
QKRYLOV_API int                     qkrylov_device_vector_dot(const qkrylov_device_vector_h x, const qkrylov_device_vector_h y, double* dot_re, double* dot_im);
QKRYLOV_API int                     qkrylov_device_vector_norm(const qkrylov_device_vector_h x, double* norm_out);
QKRYLOV_API int                     qkrylov_device_vector_axpy(double a_re, double a_im, const qkrylov_device_vector_h x, qkrylov_device_vector_h y);
QKRYLOV_API int                     qkrylov_device_vector_scal(double a_re, double a_im, qkrylov_device_vector_h x);
QKRYLOV_API int                     qkrylov_device_vector_normalize(qkrylov_device_vector_h x);
QKRYLOV_API int                     qkrylov_device_vector_zero_fill(qkrylov_device_vector_h x);
QKRYLOV_API int                     qkrylov_device_vector_copy(const qkrylov_device_vector_h src, qkrylov_device_vector_h dst);

#ifdef __cplusplus
}
#endif

#endif /* QKRYLOV_C_API_H */
