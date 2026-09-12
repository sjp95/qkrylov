#include "qkrylov/c_api.h"
#include <iostream>
#include <cmath>
#undef NDEBUG
#include <cassert>
#include <vector>
#include <complex>
#include <string>
#include <cstring>
#include <thread>

int main() {
    std::cout << "Testing Dual-Precision C API..." << std::endl;

    // 0. Test Error Diagnostics API
    std::cout << "\n--- Testing Error Diagnostics API ---" << std::endl;
    qkrylov_clear_last_error();
    assert(std::string(qkrylov_get_last_error_message()) == "");

    // Verify invalid argument error on null sector
    int err_code = qkrylov_sector_set_sz(nullptr, 0);
    assert(err_code == QKRYLOV_ERROR_INVALID_ARG);
    std::string err_msg = qkrylov_get_last_error_message();
    assert(!err_msg.empty());
    assert(err_msg.find("sector handle is null") != std::string::npos);

    // Verify error clearing
    qkrylov_clear_last_error();
    assert(std::string(qkrylov_get_last_error_message()) == "");

    // Verify invalid argument on OpSum API
    err_code = qkrylov_opsum_add_term_1body(nullptr, 1.0, 0.0, "Sz", 0);
    assert(err_code == QKRYLOV_ERROR_INVALID_ARG);
    assert(std::string(qkrylov_get_last_error_message()).find("null opsum") != std::string::npos);

    // Verify invalid argument on Hamiltonian apply
    err_code = qkrylov_hamiltonian_apply(nullptr, nullptr, nullptr, nullptr, nullptr);
    assert(err_code == QKRYLOV_ERROR_INVALID_ARG);
    assert(std::string(qkrylov_get_last_error_message()).find("hamiltonian handle is null") != std::string::npos);

    // Verify invalid argument on solvers
    err_code = qkrylov_lanczos_ground_state(nullptr, 10, 1e-6, nullptr);
    assert(err_code == QKRYLOV_ERROR_INVALID_ARG);
    assert(std::string(qkrylov_get_last_error_message()).find("hamiltonian handle is null") != std::string::npos);

    err_code = qkrylov_lanczos_two_pass_ground_state(nullptr, 10, 1e-6, nullptr);
    assert(err_code == QKRYLOV_ERROR_INVALID_ARG);
    assert(std::string(qkrylov_get_last_error_message()).find("hamiltonian handle is null") != std::string::npos);

    // Verify exception capture during Hamiltonian creation with unknown operator
    qkrylov_basis_h dummy_basis = qkrylov_spinhalf_basis_create(2, nullptr);
    qkrylov_site_h dummy_site = qkrylov_spinhalf_site_create();
    qkrylov_opsum_h bad_opsum = qkrylov_opsum_create();
    assert(qkrylov_opsum_add_term_1body(bad_opsum, 1.0, 0.0, "NONEXISTENT_OPERATOR", 0) == QKRYLOV_SUCCESS);
    qkrylov_clear_last_error();
    qkrylov_hamiltonian_h bad_H = qkrylov_hamiltonian_create(dummy_basis, dummy_site, bad_opsum);
    assert(bad_H == nullptr);
    std::string exc_msg = qkrylov_get_last_error_message();
    assert(!exc_msg.empty());
    assert(exc_msg.find("Unknown spin operator") != std::string::npos);
    std::cout << "Captured expected exception: " << exc_msg << std::endl;
    qkrylov_opsum_destroy(bad_opsum);
    qkrylov_site_destroy(dummy_site);
    qkrylov_basis_destroy(dummy_basis);
    qkrylov_clear_last_error();

    // Verify thread-local isolation of error messages
    std::thread worker([]() {
        qkrylov_clear_last_error();
        assert(std::string(qkrylov_get_last_error_message()) == "");
        int rc = qkrylov_sector_set_sz(nullptr, 0);
        assert(rc == QKRYLOV_ERROR_INVALID_ARG);
        assert(!std::string(qkrylov_get_last_error_message()).empty());
    });
    worker.join();
    // The main thread should still see empty error message
    assert(std::string(qkrylov_get_last_error_message()) == "");
    std::cout << "Error Diagnostics API verified successfully.\n" << std::endl;

    // 1. Test Sector API
    qkrylov_sector_h sec = qkrylov_sector_create();
    assert(sec != NULL);
    int res_sz = qkrylov_sector_set_sz(sec, 0);
    assert(res_sz == QKRYLOV_SUCCESS);

    // 2. Test Basis API (4-site SpinHalfBasis with total Sz = 0)
    int N = 4;
    qkrylov_basis_h basis = qkrylov_spinhalf_basis_create(N, sec);
    assert(basis != NULL);
    uint64_t dim = qkrylov_basis_dimension(basis);
    std::cout << "Basis dimension for N=4, Sz=0: " << dim << std::endl;
    assert(dim == 6); // 4 choose 2 = 6 states in Sz=0 sector
    assert(qkrylov_basis_nsites(basis) == 4);

    // Test Basis state lookups
    uint64_t s0 = qkrylov_basis_state(basis, 0);
    assert(qkrylov_basis_contains(basis, s0) == 1);
    assert(qkrylov_basis_index(basis, s0) == 0);

    // Test Basis Reflection API
    std::cout << "\n--- Testing Basis Reflection API ---" << std::endl;
    qkrylov_basis_type_t b_type;
    double b_spin = 0.0;
    int b_d = 0;
    assert(qkrylov_basis_get_type(basis, &b_type) == QKRYLOV_SUCCESS);
    assert(b_type == QKRYLOV_BASIS_SPIN_HALF);
    assert(qkrylov_basis_get_spin(basis, &b_spin) == QKRYLOV_SUCCESS);
    assert(std::abs(b_spin - 0.5) < 1e-12);
    assert(qkrylov_basis_get_dimension_per_site(basis, &b_d) == QKRYLOV_SUCCESS);
    assert(b_d == 2);

    qkrylov_sector_h cloned_sec = qkrylov_basis_get_sector(basis);
    assert(cloned_sec != nullptr);
    int sz2_val = 0, sz2_active = 0;
    assert(qkrylov_sector_get_sz(cloned_sec, &sz2_val, &sz2_active) == QKRYLOV_SUCCESS);
    assert(sz2_active == 1);
    assert(sz2_val == 0);
    qkrylov_sector_destroy(cloned_sec);

    // Test SpinSBasis Reflection
    qkrylov_basis_h s1_basis = qkrylov_basis_create_spin_s(2, 1.0, nullptr);
    assert(s1_basis != nullptr);
    assert(qkrylov_basis_get_type(s1_basis, &b_type) == QKRYLOV_SUCCESS && b_type == QKRYLOV_BASIS_SPIN_S);
    assert(qkrylov_basis_get_spin(s1_basis, &b_spin) == QKRYLOV_SUCCESS && std::abs(b_spin - 1.0) < 1e-12);
    assert(qkrylov_basis_get_dimension_per_site(s1_basis, &b_d) == QKRYLOV_SUCCESS && b_d == 3);
    qkrylov_basis_destroy(s1_basis);

    // Test FermionBasis, HubbardBasis, TJBasis Reflection
    qkrylov_basis_h ferm_b = qkrylov_fermion_basis_create(2, nullptr);
    assert(qkrylov_basis_get_type(ferm_b, &b_type) == QKRYLOV_SUCCESS && b_type == QKRYLOV_BASIS_FERMION);
    assert(qkrylov_basis_get_dimension_per_site(ferm_b, &b_d) == QKRYLOV_SUCCESS && b_d == 2);
    qkrylov_basis_destroy(ferm_b);

    qkrylov_basis_h hubb_b = qkrylov_hubbard_basis_create(2, nullptr);
    assert(qkrylov_basis_get_type(hubb_b, &b_type) == QKRYLOV_SUCCESS && b_type == QKRYLOV_BASIS_HUBBARD);
    assert(qkrylov_basis_get_dimension_per_site(hubb_b, &b_d) == QKRYLOV_SUCCESS && b_d == 4);
    qkrylov_basis_destroy(hubb_b);

    qkrylov_basis_h tj_b = qkrylov_tj_basis_create(2, nullptr);
    assert(qkrylov_basis_get_type(tj_b, &b_type) == QKRYLOV_SUCCESS && b_type == QKRYLOV_BASIS_TJ);
    assert(qkrylov_basis_get_dimension_per_site(tj_b, &b_d) == QKRYLOV_SUCCESS && b_d == 3);
    qkrylov_basis_destroy(tj_b);
    std::cout << "Basis Reflection API verified successfully." << std::endl;

    // Test Sector set_n and set_nb functions
    assert(qkrylov_sector_set_n(sec, 2) == QKRYLOV_SUCCESS);
    assert(qkrylov_sector_set_nb(sec, 1) == QKRYLOV_SUCCESS);

    // 3. Test Site API & Site Reflection & Site::apply
    std::cout << "\n--- Testing Site API & Local Action Evaluation ---" << std::endl;
    qkrylov_site_h site = qkrylov_spinhalf_site_create();
    assert(site != NULL);

    qkrylov_site_type_t s_type;
    double s_spin = 0.0;
    int s_d = 0;
    assert(qkrylov_site_get_type(site, &s_type) == QKRYLOV_SUCCESS && s_type == QKRYLOV_SITE_SPIN_HALF);
    assert(qkrylov_site_get_spin(site, &s_spin) == QKRYLOV_SUCCESS && std::abs(s_spin - 0.5) < 1e-12);
    assert(qkrylov_site_get_dimension_per_site(site, &s_d) == QKRYLOV_SUCCESS && s_d == 2);

    // Test Site::apply on SpinHalf
    qkrylov_local_action_t act;
    // Sz on |1> (spin-up, site 0) -> +0.5
    assert(qkrylov_site_apply(site, "Sz", 0, 1ULL, &act) == QKRYLOV_SUCCESS);
    assert(act.valid == 1 && act.new_state == 1ULL && std::abs(act.matrix_element_re - 0.5) < 1e-12);

    // Sz on |0> (spin-down, site 0) -> -0.5
    assert(qkrylov_site_apply(site, "Sz", 0, 0ULL, &act) == QKRYLOV_SUCCESS);
    assert(act.valid == 1 && act.new_state == 0ULL && std::abs(act.matrix_element_re - (-0.5)) < 1e-12);

    // Sp on |0> -> |1> with element 1.0
    assert(qkrylov_site_apply(site, "Sp", 0, 0ULL, &act) == QKRYLOV_SUCCESS);
    assert(act.valid == 1 && act.new_state == 1ULL && std::abs(act.matrix_element_re - 1.0) < 1e-12);

    // Sp on |1> -> annihilated (valid = 0)
    assert(qkrylov_site_apply(site, "Sp", 0, 1ULL, &act) == QKRYLOV_SUCCESS);
    assert(act.valid == 0);

    // Sm on |1> -> |0> with element 1.0
    assert(qkrylov_site_apply(site, "Sm", 0, 1ULL, &act) == QKRYLOV_SUCCESS);
    assert(act.valid == 1 && act.new_state == 0ULL && std::abs(act.matrix_element_re - 1.0) < 1e-12);

    // Sx on |0> -> |1> with element 0.5
    assert(qkrylov_site_apply(site, "Sx", 0, 0ULL, &act) == QKRYLOV_SUCCESS);
    assert(act.valid == 1 && act.new_state == 1ULL && std::abs(act.matrix_element_re - 0.5) < 1e-12);

    // Sy on |0> -> |1> with element +0.5i
    assert(qkrylov_site_apply(site, "Sy", 0, 0ULL, &act) == QKRYLOV_SUCCESS);
    assert(act.valid == 1 && act.new_state == 1ULL && std::abs(act.matrix_element_im - 0.5) < 1e-12);

    // Invalid operator on Site::apply
    assert(qkrylov_site_apply(site, "UNKNOWN_OP", 0, 0ULL, &act) == QKRYLOV_ERROR_EXCEPTION);
    assert(!std::string(qkrylov_get_last_error_message()).empty());
    qkrylov_clear_last_error();

    // Test SpinSSite reflection & apply (S=1)
    qkrylov_site_h s1_site = qkrylov_site_create_spin_s(1.0);
    assert(s1_site != nullptr);
    assert(qkrylov_site_get_type(s1_site, &s_type) == QKRYLOV_SUCCESS && s_type == QKRYLOV_SITE_SPIN_S);
    assert(qkrylov_site_get_spin(s1_site, &s_spin) == QKRYLOV_SUCCESS && std::abs(s_spin - 1.0) < 1e-12);
    assert(qkrylov_site_get_dimension_per_site(s1_site, &s_d) == QKRYLOV_SUCCESS && s_d == 3);

    // S=1, site 0, state 0 (m_z = -1): Sp -> state 1 (m_z = 0), matrix element = sqrt(2)
    assert(qkrylov_site_apply(s1_site, "Sp", 0, 0ULL, &act) == QKRYLOV_SUCCESS);
    assert(act.valid == 1 && act.new_state == 1ULL && std::abs(act.matrix_element_re - std::sqrt(2.0)) < 1e-12);
    qkrylov_site_destroy(s1_site);

    // Test FermionSite reflection & apply
    qkrylov_site_h ferm_site = qkrylov_fermion_site_create();
    assert(ferm_site != nullptr);
    assert(qkrylov_site_get_type(ferm_site, &s_type) == QKRYLOV_SUCCESS && s_type == QKRYLOV_SITE_FERMION);
    assert(qkrylov_site_get_dimension_per_site(ferm_site, &s_d) == QKRYLOV_SUCCESS && s_d == 2);
    // N on |1> -> valid=1, me=1.0
    assert(qkrylov_site_apply(ferm_site, "N", 0, 1ULL, &act) == QKRYLOV_SUCCESS);
    assert(act.valid == 1 && act.new_state == 1ULL && std::abs(act.matrix_element_re - 1.0) < 1e-12);
    // C on |1> -> valid=1, new_state=0, me=1.0
    assert(qkrylov_site_apply(ferm_site, "C", 0, 1ULL, &act) == QKRYLOV_SUCCESS);
    assert(act.valid == 1 && act.new_state == 0ULL && std::abs(act.matrix_element_re - 1.0) < 1e-12);
    // Cdag on |1> -> valid=0
    assert(qkrylov_site_apply(ferm_site, "Cdag", 0, 1ULL, &act) == QKRYLOV_SUCCESS);
    assert(act.valid == 0);
    qkrylov_site_destroy(ferm_site);
    std::cout << "Site API & Local Action Evaluation verified successfully." << std::endl;

    // 4. Test OpSum API (4-site Heisenberg chain)
    std::cout << "\n--- Testing OpSum API & OpSum Reflection ---" << std::endl;
    qkrylov_opsum_h opsum = qkrylov_opsum_create();
    assert(opsum != NULL);

    for (int i = 0; i < N - 1; ++i) {
        // Sz_i Sz_{i+1}
        assert(qkrylov_opsum_add_term_2body(opsum, 1.0, 0.0, "Sz", i, "Sz", i+1) == QKRYLOV_SUCCESS);
        // 0.5 Sp_i Sm_{i+1}
        assert(qkrylov_opsum_add_term_2body(opsum, 0.5, 0.0, "Sp", i, "Sm", i+1) == QKRYLOV_SUCCESS);
        // 0.5 Sm_i Sp_{i+1}
        assert(qkrylov_opsum_add_term_2body(opsum, 0.5, 0.0, "Sm", i, "Sp", i+1) == QKRYLOV_SUCCESS);
    }

    // Verify OpSum size on Heisenberg chain (3 bonds * 3 terms = 9 terms)
    int n_op_terms = 0;
    assert(qkrylov_opsum_size(opsum, &n_op_terms) == QKRYLOV_SUCCESS);
    assert(n_op_terms == 9);

    // Test 3-Body N-Body Term API on separate OpSum handle + OpSum term inspection
    qkrylov_opsum_h opsum_nbody = qkrylov_opsum_create();
    assert(opsum_nbody != NULL);
    const char* ops3[3] = {"Sz", "Sz", "Sz"};
    int sites3[3] = {0, 1, 2};
    assert(qkrylov_opsum_add_term_nbody(opsum_nbody, 0.1, 0.2, 3, ops3, sites3) == QKRYLOV_SUCCESS);

    int nb_size = 0;
    assert(qkrylov_opsum_size(opsum_nbody, &nb_size) == QKRYLOV_SUCCESS);
    assert(nb_size == 1);

    double c_re = 0.0, c_im = 0.0;
    int n_factors = 0;
    assert(qkrylov_opsum_get_term_info(opsum_nbody, 0, &c_re, &c_im, &n_factors) == QKRYLOV_SUCCESS);
    assert(std::abs(c_re - 0.1) < 1e-12);
    assert(std::abs(c_im - 0.2) < 1e-12);
    assert(n_factors == 3);

    for (int k = 0; k < 3; ++k) {
        char op_name[16] = {0};
        int f_site = -1;
        assert(qkrylov_opsum_get_factor(opsum_nbody, 0, k, op_name, sizeof(op_name), &f_site) == QKRYLOV_SUCCESS);
        assert(std::string(op_name) == "Sz");
        assert(f_site == k);
    }

    // Out-of-bounds checks
    assert(qkrylov_opsum_get_term_info(opsum_nbody, 1, &c_re, &c_im, &n_factors) == QKRYLOV_ERROR_INVALID_ARG);
    assert(qkrylov_opsum_get_factor(opsum_nbody, 0, 5, nullptr, 0, nullptr) == QKRYLOV_ERROR_INVALID_ARG);

    qkrylov_opsum_destroy(opsum_nbody);
    std::cout << "OpSum API & Reflection verified successfully." << std::endl;

    // 5. Test Device & Hardware Query API
    int is_gpu = qkrylov_is_gpu_build();
    int gpus = qkrylov_gpu_count();
    const char* gpu_backend = qkrylov_find_gpu();
    std::cout << "Device check: is_gpu=" << is_gpu << ", gpu_count=" << gpus
              << ", backend=" << (gpu_backend ? gpu_backend : "none") << std::endl;
    assert(qkrylov_initialize_device("cpu") == QKRYLOV_SUCCESS);

    // =========================================================================
    // PART A: Double Precision (FP64) C API Verification
    // =========================================================================
    std::cout << "\n--- Testing FP64 C API Pipeline ---" << std::endl;
    qkrylov_hamiltonian_h H = qkrylov_hamiltonian_create(basis, site, opsum);
    assert(H != NULL);
    assert(qkrylov_hamiltonian_dimension(H) == dim);
    assert(qkrylov_hamiltonian_precision(H) == QKRYLOV_PRECISION_FP64);

    // Test Matrix-Vector Apply (FP64)
    std::vector<double> x_real(dim, 1.0);
    std::vector<double> x_imag(dim, 0.0);
    std::vector<double> y_real(dim, 0.0);
    std::vector<double> y_imag(dim, 0.0);

    int apply_res = qkrylov_hamiltonian_apply(H, x_real.data(), x_imag.data(), y_real.data(), y_imag.data());
    assert(apply_res == QKRYLOV_SUCCESS);

    // Test Zero-Copy Direct Complex Apply (FP64)
    std::vector<std::complex<double>> x_cx(dim, std::complex<double>(1.0, 0.0));
    std::vector<std::complex<double>> y_cx(dim, std::complex<double>(0.0, 0.0));
    int apply_cx_res = qkrylov_hamiltonian_apply_complex(H, reinterpret_cast<const double*>(x_cx.data()), reinterpret_cast<double*>(y_cx.data()));
    assert(apply_cx_res == QKRYLOV_SUCCESS);
    for (size_t i = 0; i < dim; ++i) {
        assert(std::abs(y_cx[i].real() - y_real[i]) < 1e-12);
        assert(std::abs(y_cx[i].imag() - y_imag[i]) < 1e-12);
    }

    // Test Matrix-Free Diagonal Extraction (FP64)
    std::vector<double> diag(dim);
    int diag_res = qkrylov_hamiltonian_diagonal(H, diag.data());
    assert(diag_res == QKRYLOV_SUCCESS);

    // Test Lanczos Ground State Solver via FP64 C API (random start with nullptr)
    qkrylov_lanczos_result_c_t lanczos_res;
    std::vector<std::complex<double>> psi_cx(dim);
    int solver_res = qkrylov_lanczos_ground_state_complex(H, 200, 1e-12, &lanczos_res, reinterpret_cast<double*>(psi_cx.data()));
    assert(solver_res == QKRYLOV_SUCCESS);
    assert(lanczos_res.converged == 1);

    std::cout << "C API Lanczos FP64 Ground State Energy: " << lanczos_res.energy << std::endl;
    // Exact Heisenberg N=4 ground state energy is -1.6160254037844386
    assert(std::abs(lanczos_res.energy - (-1.6160254037844386)) < 1e-10);

    // Test Two-Pass Lanczos Ground State Solver via FP64 C API
    qkrylov_lanczos_result_c_t tp_res;
    std::vector<std::complex<double>> tp_psi(dim);
    int tp_status = qkrylov_lanczos_two_pass_ground_state_complex(H, 200, 1e-12, &tp_res, reinterpret_cast<double*>(tp_psi.data()));
    assert(tp_status == QKRYLOV_SUCCESS);
    assert(tp_res.converged == 1);
    assert(std::abs(tp_res.energy - lanczos_res.energy) < 1e-10);

    // Verify eigenvector normalization: ||psi||^2 == 1.0
    double norm_sq = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        norm_sq += std::norm(psi_cx[i]);
    }
    assert(std::abs(norm_sq - 1.0) < 1e-12);

    // Test Davidson Solver via FP64 C API (Lowest 2 Eigenpairs)
    int n_eig = 2;
    std::vector<double> dav_evals(n_eig);
    std::vector<std::complex<double>> dav_evecs(n_eig * dim);
    qkrylov_davidson_result_c_t dav_info;
    int dav_status = qkrylov_davidson_lowest_complex(H, n_eig, 20, 1e-10, dav_evals.data(), reinterpret_cast<double*>(dav_evecs.data()), &dav_info);
    assert(dav_status == QKRYLOV_SUCCESS);
    assert(dav_info.converged == 1);

    std::cout << "C API Davidson FP64 Lowest Eigenvalues: E0=" << dav_evals[0] << ", E1=" << dav_evals[1] << std::endl;
    assert(std::abs(dav_evals[0] - lanczos_res.energy) < 1e-10);
    assert(dav_evals[0] <= dav_evals[1]);

    // Verify H * psi_k == E_k * psi_k
    for (int k = 0; k < n_eig; ++k) {
        const std::complex<double>* vk = dav_evecs.data() + (k * dim);
        std::vector<std::complex<double>> Hvk(dim);
        assert(qkrylov_hamiltonian_apply_complex(H, reinterpret_cast<const double*>(vk), reinterpret_cast<double*>(Hvk.data())) == QKRYLOV_SUCCESS);
        for (size_t i = 0; i < dim; ++i) {
            std::complex<double> expected = dav_evals[k] * vk[i];
            assert(std::abs(Hvk[i] - expected) < 1e-8);
        }
    }

    // Test Dynamics & Spectral Function (FP64)
    int n_iter = 20;
    std::vector<double> alphas(n_iter);
    std::vector<double> betas(n_iter);
    double norm_phi0 = 0.0;
    int num_coeffs = 0;
    int dyn_status = qkrylov_continued_fraction_coeffs_complex(H, reinterpret_cast<const double*>(psi_cx.data()), n_iter, alphas.data(), betas.data(), &norm_phi0, &num_coeffs);
    assert(dyn_status == QKRYLOV_SUCCESS);
    assert(num_coeffs > 0);
    assert(std::abs(norm_phi0 - 1.0) < 1e-10);

    double spec_val = qkrylov_evaluate_spectral_function(alphas.data(), betas.data(), num_coeffs, norm_phi0, 0.5, lanczos_res.energy, 0.1);
    assert(spec_val > 0.0);

    // Test FTLM (FP64)
    qkrylov_ftlm_result_c_t ftlm_res;
    int ftlm_status = qkrylov_ftlm(H, 1.0, 10, 20, &ftlm_res);
    assert(ftlm_status == QKRYLOV_SUCCESS);
    assert(ftlm_res.partition_function > 0.0);

    // Test FTLM Sweep & Observables (FP64)
    double betas_sweep[3] = {0.5, 1.0, 2.0};
    qkrylov_hamiltonian_h obs_arr[1] = {H};
    qkrylov_ftlm_sweep_result_fp64_t sweep_res64;
    int sweep_status = qkrylov_ftlm_sweep_fp64(H, betas_sweep, 3, obs_arr, 1, 20, 10, 42, &sweep_res64);
    assert(sweep_status == QKRYLOV_SUCCESS);
    assert(sweep_res64.num_betas == 3);
    assert(sweep_res64.num_observables == 1);
    assert(sweep_res64.partition_functions[0] > 0.0);
    assert(std::abs(sweep_res64.observable_expectations[0] - sweep_res64.internal_energies[0]) < 1e-4);
    qkrylov_ftlm_sweep_result_free_fp64(&sweep_res64);

    // =========================================================================
    // PART B: Single Precision (FP32) C API Verification
    // =========================================================================
    std::cout << "\n--- Testing FP32 C API Pipeline ---" << std::endl;
    // Pass the EXACT SAME basis, site, and opsum handles!
    qkrylov_hamiltonian_h H32 = qkrylov_hamiltonian_create_fp32(basis, site, opsum);
    assert(H32 != NULL);
    assert(qkrylov_hamiltonian_dimension(H32) == dim);
    assert(qkrylov_hamiltonian_precision(H32) == QKRYLOV_PRECISION_FP32);

    // Test Matrix-Vector Apply (FP32)
    std::vector<float> x_real32(dim, 1.0f);
    std::vector<float> x_imag32(dim, 0.0f);
    std::vector<float> y_real32(dim, 0.0f);
    std::vector<float> y_imag32(dim, 0.0f);
    int apply_res32 = qkrylov_hamiltonian_apply_fp32(H32, x_real32.data(), x_imag32.data(), y_real32.data(), y_imag32.data());
    assert(apply_res32 == QKRYLOV_SUCCESS);

    // Test Zero-Copy Direct Complex Apply (FP32)
    std::vector<std::complex<float>> x_cx32(dim, std::complex<float>(1.0f, 0.0f));
    std::vector<std::complex<float>> y_cx32(dim, std::complex<float>(0.0f, 0.0f));
    int apply_cx_res32 = qkrylov_hamiltonian_apply_complex_fp32(H32, reinterpret_cast<const float*>(x_cx32.data()), reinterpret_cast<float*>(y_cx32.data()));
    assert(apply_cx_res32 == QKRYLOV_SUCCESS);
    for (size_t i = 0; i < dim; ++i) {
        assert(std::abs(y_cx32[i].real() - y_real32[i]) < 1e-5f);
        assert(std::abs(y_cx32[i].imag() - y_imag32[i]) < 1e-5f);
    }

    // Test Lanczos Ground State (FP32)
    qkrylov_lanczos_result_fp32_t lanczos_res32;
    std::vector<std::complex<float>> psi_cx32(dim);
    int solver_res32 = qkrylov_lanczos_ground_state_complex_fp32(H32, 200, 1e-5f, &lanczos_res32, reinterpret_cast<float*>(psi_cx32.data()));
    assert(solver_res32 == QKRYLOV_SUCCESS);
    std::cout << "C API Lanczos FP32 Ground State Energy: " << lanczos_res32.energy << std::endl;
    assert(lanczos_res32.converged == 1 || lanczos_res32.iterations == static_cast<int>(dim));
    assert(std::abs(lanczos_res32.energy - (-1.6160254038f)) < 1e-4f);

    // Test Two-Pass Lanczos Ground State (FP32)
    qkrylov_lanczos_result_fp32_t tp_res32;
    int tp_status32 = qkrylov_lanczos_two_pass_ground_state_fp32(H32, 200, 1e-5f, &tp_res32);
    assert(tp_status32 == QKRYLOV_SUCCESS);
    assert(tp_res32.converged == 1 || tp_res32.iterations == static_cast<int>(dim));
    assert(std::abs(tp_res32.energy - (-1.6160254038f)) < 1e-4f);

    // Test Davidson Lowest (FP32)
    std::vector<float> dav_evals32(n_eig);
    std::vector<std::complex<float>> dav_evecs32(n_eig * dim);
    qkrylov_davidson_result_c_t dav_info32;
    int dav_status32 = qkrylov_davidson_lowest_complex_fp32(H32, n_eig, 20, 1e-5f, dav_evals32.data(), reinterpret_cast<float*>(dav_evecs32.data()), &dav_info32);
    assert(dav_status32 == QKRYLOV_SUCCESS);
    assert(dav_info32.converged == 1);
    assert(std::abs(dav_evals32[0] - lanczos_res32.energy) < 1e-4f);

    // Test Precision Mismatch Protection
    assert(qkrylov_hamiltonian_apply_fp32(H, x_real32.data(), x_imag32.data(), y_real32.data(), y_imag32.data()) == QKRYLOV_ERROR_INVALID_ARG);
    assert(qkrylov_hamiltonian_apply_fp64(H32, x_real.data(), x_imag.data(), y_real.data(), y_imag.data()) == QKRYLOV_ERROR_INVALID_ARG);

    // Test FTLM Sweep & Observables (FP32)
    float betas_sweep32[2] = {1.0f, 2.0f};
    qkrylov_hamiltonian_h obs_arr32[1] = {H32};
    qkrylov_ftlm_sweep_result_fp32_t sweep_res32;
    int sweep32_status = qkrylov_ftlm_sweep_fp32(H32, betas_sweep32, 2, obs_arr32, 1, 20, 10, 42, &sweep_res32);
    assert(sweep32_status == QKRYLOV_SUCCESS);
    assert(sweep_res32.num_betas == 2);
    assert(sweep_res32.num_observables == 1);
    assert(sweep_res32.partition_functions[0] > 0.0f);
    qkrylov_ftlm_sweep_result_free_fp32(&sweep_res32);

    // Cleanup Hamiltonians
    qkrylov_hamiltonian_destroy(H);
    qkrylov_hamiltonian_destroy(H32);
    qkrylov_opsum_destroy(opsum);
    qkrylov_site_destroy(site);
    qkrylov_basis_destroy(basis);
    qkrylov_sector_destroy(sec);

    // =========================================================================
    // PART C: Spin-1 and Correction Vector Testing
    // =========================================================================
    std::cout << "\n--- Testing Spin-1 and Correction Vector ---" << std::endl;
    qkrylov_site_h spin1_site = qkrylov_site_create_spin_s(1.0);
    assert(spin1_site != NULL);

    qkrylov_sector_h sec_spin1 = qkrylov_sector_create();
    assert(sec_spin1 != NULL);
    assert(qkrylov_sector_set_sz(sec_spin1, 0) == QKRYLOV_SUCCESS);

    int N_s1 = 4;
    qkrylov_basis_h spin1_basis = qkrylov_basis_create_spin_s(N_s1, 1.0, sec_spin1);
    assert(spin1_basis != NULL);
    uint64_t dim_s1 = qkrylov_basis_dimension(spin1_basis);
    std::cout << "Spin-1 Basis dimension for N=4, Sz=0: " << dim_s1 << std::endl;
    assert(dim_s1 == 19);

    qkrylov_opsum_h ops_s1 = qkrylov_opsum_create();
    assert(ops_s1 != NULL);
    for (int i = 0; i < N_s1 - 1; ++i) {
        assert(qkrylov_opsum_add_term_2body(ops_s1, 1.0, 0.0, "Sz", i, "Sz", i+1) == QKRYLOV_SUCCESS);
        assert(qkrylov_opsum_add_term_2body(ops_s1, 0.5, 0.0, "Sp", i, "Sm", i+1) == QKRYLOV_SUCCESS);
        assert(qkrylov_opsum_add_term_2body(ops_s1, 0.5, 0.0, "Sm", i, "Sp", i+1) == QKRYLOV_SUCCESS);
    }

    qkrylov_hamiltonian_h H_s1 = qkrylov_hamiltonian_create(spin1_basis, spin1_site, ops_s1);
    assert(H_s1 != NULL);

    qkrylov_lanczos_result_c_t lanczos_s1_res;
    std::vector<std::complex<double>> psi0_s1(dim_s1);
    int gs_status = qkrylov_lanczos_ground_state_complex(
        H_s1, 200, 1e-10, &lanczos_s1_res, reinterpret_cast<double*>(psi0_s1.data())
    );
    assert(gs_status == QKRYLOV_SUCCESS);
    assert(lanczos_s1_res.converged == 1);
    std::cout << "Spin-1 N=4 FP64 Ground State Energy: " << lanczos_s1_res.energy << std::endl;

    // Apply local operator O = Sz_0
    qkrylov_opsum_h ops_sz0 = qkrylov_opsum_create();
    assert(ops_sz0 != NULL);
    assert(qkrylov_opsum_add_term_1body(ops_sz0, 1.0, 0.0, "Sz", 0) == QKRYLOV_SUCCESS);
    qkrylov_hamiltonian_h H_sz0 = qkrylov_hamiltonian_create(spin1_basis, spin1_site, ops_sz0);
    assert(H_sz0 != NULL);

    std::vector<std::complex<double>> op_psi0(dim_s1);
    int apply_sz0_status = qkrylov_hamiltonian_apply_complex(
        H_sz0,
        reinterpret_cast<const double*>(psi0_s1.data()),
        reinterpret_cast<double*>(op_psi0.data())
    );
    assert(apply_sz0_status == QKRYLOV_SUCCESS);

    // Correction Vector Spectroscopy Solver (FP64)
    qkrylov_correction_vector_result_c_t cv_result;
    std::vector<std::complex<double>> cv_vec_out(dim_s1);
    double omega = 1.5;
    double eta = 0.1;
    int cv_status = qkrylov_solver_correction_vector(
        H_s1,
        reinterpret_cast<const double*>(op_psi0.data()),
        lanczos_s1_res.energy,
        omega,
        eta,
        500,
        1e-8,
        &cv_result,
        reinterpret_cast<double*>(cv_vec_out.data())
    );
    assert(cv_status == QKRYLOV_SUCCESS);
    assert(cv_result.converged == 1);
    std::cout << "C API Correction Vector FP64 Result: S(w=" << omega << ")=" << cv_result.spectral_function << std::endl;

    // Correction Vector Spectroscopy Solver (FP32)
    qkrylov_hamiltonian_h H_s1_32 = qkrylov_hamiltonian_create_fp32(spin1_basis, spin1_site, ops_s1);
    assert(H_s1_32 != NULL);
    std::vector<std::complex<float>> op_psi0_32(dim_s1);
    for (size_t i = 0; i < dim_s1; ++i) {
        op_psi0_32[i] = std::complex<float>(static_cast<float>(op_psi0[i].real()), static_cast<float>(op_psi0[i].imag()));
    }
    qkrylov_correction_vector_result_fp32_t cv_result32;
    int cv_status32 = qkrylov_solver_correction_vector_fp32(
        H_s1_32,
        reinterpret_cast<const float*>(op_psi0_32.data()),
        static_cast<float>(lanczos_s1_res.energy),
        static_cast<float>(omega),
        static_cast<float>(eta),
        500,
        1e-5f,
        &cv_result32,
        nullptr
    );
    assert(cv_status32 == QKRYLOV_SUCCESS);
    assert(cv_result32.converged == 1);
    assert(std::abs(cv_result32.spectral_function - static_cast<float>(cv_result.spectral_function)) < 1e-4f);

    // Cleanup Spin-1 handles
    qkrylov_hamiltonian_destroy(H_sz0);
    qkrylov_opsum_destroy(ops_sz0);
    qkrylov_hamiltonian_destroy(H_s1);
    qkrylov_hamiltonian_destroy(H_s1_32);
    qkrylov_opsum_destroy(ops_s1);
    qkrylov_site_destroy(spin1_site);
    qkrylov_basis_destroy(spin1_basis);
    qkrylov_sector_destroy(sec_spin1);

    // =========================================================================
    // PART D: Kokkos Parallel Vector Operations (BLAS-1 Kernels)
    // =========================================================================
    std::cout << "\n--- Testing Parallel Vector Operations (BLAS-1) ---" << std::endl;

    // 1. Test Dot Product (FP64 & FP32)
    // x = [1 + 2i, 3 + 4i, -1 + 0i, 0 - 2i, 2 + 1i]
    // y = [2 - 1i, 0 + 3i, 4 + 1i, 1 - 1i, -2 + 0i]
    // <x|y> = sum_i conj(x_i) * y_i = 6 + 7i
    uint64_t vdim = 5;
    std::vector<std::complex<double>> vx64 = {
        {1.0, 2.0}, {3.0, 4.0}, {-1.0, 0.0}, {0.0, -2.0}, {2.0, 1.0}
    };
    std::vector<std::complex<double>> vy64 = {
        {2.0, -1.0}, {0.0, 3.0}, {4.0, 1.0}, {1.0, -1.0}, {-2.0, 0.0}
    };
    double dot_re64 = 0.0, dot_im64 = 0.0;
    assert(qkrylov_vector_dot_fp64(vdim,
        reinterpret_cast<const double*>(vx64.data()),
        reinterpret_cast<const double*>(vy64.data()),
        &dot_re64, &dot_im64) == QKRYLOV_SUCCESS);
    assert(std::abs(dot_re64 - 6.0) < 1e-12);
    assert(std::abs(dot_im64 - 7.0) < 1e-12);

    // Default alias test
    double dot_re_alias = 0.0, dot_im_alias = 0.0;
    assert(qkrylov_vector_dot(vdim,
        reinterpret_cast<const double*>(vx64.data()),
        reinterpret_cast<const double*>(vy64.data()),
        &dot_re_alias, &dot_im_alias) == QKRYLOV_SUCCESS);
    assert(std::abs(dot_re_alias - 6.0) < 1e-12);
    assert(std::abs(dot_im_alias - 7.0) < 1e-12);

    // FP32 dot product
    std::vector<std::complex<float>> vx32 = {
        {1.0f, 2.0f}, {3.0f, 4.0f}, {-1.0f, 0.0f}, {0.0f, -2.0f}, {2.0f, 1.0f}
    };
    std::vector<std::complex<float>> vy32 = {
        {2.0f, -1.0f}, {0.0f, 3.0f}, {4.0f, 1.0f}, {1.0f, -1.0f}, {-2.0f, 0.0f}
    };
    float dot_re32 = 0.0f, dot_im32 = 0.0f;
    assert(qkrylov_vector_dot_fp32(vdim,
        reinterpret_cast<const float*>(vx32.data()),
        reinterpret_cast<const float*>(vy32.data()),
        &dot_re32, &dot_im32) == QKRYLOV_SUCCESS);
    assert(std::abs(dot_re32 - 6.0f) < 1e-5f);
    assert(std::abs(dot_im32 - 7.0f) < 1e-5f);

    // 2. Test Norm (FP64 & FP32)
    // x = [3 + 4i, 0 + 0i], ||x|| = 5.0
    std::vector<std::complex<double>> vnorm_x64 = {{3.0, 4.0}, {0.0, 0.0}};
    double n64 = 0.0;
    assert(qkrylov_vector_norm_fp64(2, reinterpret_cast<const double*>(vnorm_x64.data()), &n64) == QKRYLOV_SUCCESS);
    assert(std::abs(n64 - 5.0) < 1e-12);

    float n32 = 0.0f;
    std::vector<std::complex<float>> vnorm_x32 = {{3.0f, 4.0f}, {0.0f, 0.0f}};
    assert(qkrylov_vector_norm_fp32(2, reinterpret_cast<const float*>(vnorm_x32.data()), &n32) == QKRYLOV_SUCCESS);
    assert(std::abs(n32 - 5.0f) < 1e-5f);

    // 3. Test AXPY (FP64 & FP32)
    // y = a * x + y with a = 2 + 0i, x = [1+1i, 2+0i], y = [3-1i, 1+2i] -> y = [5+1i, 5+2i]
    std::vector<std::complex<double>> axpy_x64 = {{1.0, 1.0}, {2.0, 0.0}};
    std::vector<std::complex<double>> axpy_y64 = {{3.0, -1.0}, {1.0, 2.0}};
    assert(qkrylov_vector_axpy_fp64(2, 2.0, 0.0,
        reinterpret_cast<const double*>(axpy_x64.data()),
        reinterpret_cast<double*>(axpy_y64.data())) == QKRYLOV_SUCCESS);
    assert(std::abs(axpy_y64[0].real() - 5.0) < 1e-12);
    assert(std::abs(axpy_y64[0].imag() - 1.0) < 1e-12);
    assert(std::abs(axpy_y64[1].real() - 5.0) < 1e-12);
    assert(std::abs(axpy_y64[1].imag() - 2.0) < 1e-12);

    // 4. Test SCAL (FP64 & FP32)
    // x = [2+3i, -1+4i], a = 0 + 2i -> x = [-6+4i, -8-2i]
    std::vector<std::complex<double>> scal_x64 = {{2.0, 3.0}, {-1.0, 4.0}};
    assert(qkrylov_vector_scal_fp64(2, 0.0, 2.0, reinterpret_cast<double*>(scal_x64.data())) == QKRYLOV_SUCCESS);
    assert(std::abs(scal_x64[0].real() - (-6.0)) < 1e-12);
    assert(std::abs(scal_x64[0].imag() - 4.0) < 1e-12);
    assert(std::abs(scal_x64[1].real() - (-8.0)) < 1e-12);
    assert(std::abs(scal_x64[1].imag() - (-2.0)) < 1e-12);

    // 5. Test Normalize (FP64 & FP32)
    // x = [3 + 0i, 4 + 0i], ||x|| = 5 -> normalized x = [0.6, 0.8]
    std::vector<std::complex<double>> nrm_x64 = {{3.0, 0.0}, {4.0, 0.0}};
    assert(qkrylov_vector_normalize_fp64(2, reinterpret_cast<double*>(nrm_x64.data())) == QKRYLOV_SUCCESS);
    assert(std::abs(nrm_x64[0].real() - 0.6) < 1e-12);
    assert(std::abs(nrm_x64[1].real() - 0.8) < 1e-12);
    double unit_norm = 0.0;
    assert(qkrylov_vector_norm_fp64(2, reinterpret_cast<const double*>(nrm_x64.data()), &unit_norm) == QKRYLOV_SUCCESS);
    assert(std::abs(unit_norm - 1.0) < 1e-12);

    // 6. Test Zero Fill (FP64)
    assert(qkrylov_vector_zero_fill_fp64(2, reinterpret_cast<double*>(nrm_x64.data())) == QKRYLOV_SUCCESS);
    assert(nrm_x64[0] == std::complex<double>(0.0, 0.0));
    assert(nrm_x64[1] == std::complex<double>(0.0, 0.0));

    // 7. Test Copy (FP64)
    std::vector<std::complex<double>> copy_dst64(2);
    std::vector<std::complex<double>> copy_src64 = {{7.5, -2.5}, {-3.1, 4.2}};
    assert(qkrylov_vector_copy_fp64(2,
        reinterpret_cast<const double*>(copy_src64.data()),
        reinterpret_cast<double*>(copy_dst64.data())) == QKRYLOV_SUCCESS);
    assert(copy_dst64[0] == copy_src64[0]);
    assert(copy_dst64[1] == copy_src64[1]);

    // 8. Test Error Handling (Null pointers, zero dim)
    assert(qkrylov_vector_dot_fp64(0, nullptr, nullptr, &dot_re64, &dot_im64) == QKRYLOV_SUCCESS);
    assert(dot_re64 == 0.0 && dot_im64 == 0.0);
    assert(qkrylov_vector_dot_fp64(10, nullptr, vy64.data() ? reinterpret_cast<const double*>(vy64.data()) : nullptr, &dot_re64, &dot_im64) == QKRYLOV_ERROR_INVALID_ARG);
    assert(qkrylov_vector_norm_fp64(10, nullptr, &n64) == QKRYLOV_ERROR_INVALID_ARG);
    qkrylov_clear_last_error();

    std::cout << "Parallel Vector Operations verified successfully." << std::endl;

    // =========================================================================
    // PART E: Device-Resident Vectors & Zero-Copy GPU SpMV
    // =========================================================================
    std::cout << "\n--- Testing Device-Resident Vectors & Zero-Copy SpMV ---" << std::endl;

    // 1. Setup Hamiltonian (Heisenberg 2-site)
    qkrylov_basis_h b_dev = qkrylov_spinhalf_basis_create(2, nullptr);
    qkrylov_site_h s_dev = qkrylov_spinhalf_site_create();
    qkrylov_opsum_h ops_dev = qkrylov_opsum_create();
    assert(qkrylov_opsum_add_term_2body(ops_dev, 1.0, 0.0, "Sz", 0, "Sz", 1) == QKRYLOV_SUCCESS);
    assert(qkrylov_opsum_add_term_2body(ops_dev, 0.5, 0.0, "Sp", 0, "Sm", 1) == QKRYLOV_SUCCESS);
    assert(qkrylov_opsum_add_term_2body(ops_dev, 0.5, 0.0, "Sm", 0, "Sp", 1) == QKRYLOV_SUCCESS);

    qkrylov_hamiltonian_h H_dev64 = qkrylov_hamiltonian_create_fp64(b_dev, s_dev, ops_dev);
    qkrylov_hamiltonian_h H_dev32 = qkrylov_hamiltonian_create_fp32(b_dev, s_dev, ops_dev);
    assert(H_dev64 != nullptr && H_dev32 != nullptr);
    uint64_t H_dim = qkrylov_hamiltonian_dimension(H_dev64);
    assert(H_dim == 4);

    // 2. Device Vector Allocation & Metadata (FP64 & FP32)
    qkrylov_device_vector_h x_dev64 = qkrylov_device_vector_create_fp64(H_dim);
    qkrylov_device_vector_h y_dev64 = qkrylov_device_vector_create_fp64(H_dim);
    assert(x_dev64 != nullptr && y_dev64 != nullptr);
    assert(qkrylov_device_vector_dimension(x_dev64) == 4);
    assert(qkrylov_device_vector_precision(x_dev64) == 1);
    assert(qkrylov_device_vector_data(x_dev64) != nullptr);

    qkrylov_device_vector_h x_dev32 = qkrylov_device_vector_create_fp32(H_dim);
    qkrylov_device_vector_h y_dev32 = qkrylov_device_vector_create_fp32(H_dim);
    assert(x_dev32 != nullptr && y_dev32 != nullptr);
    assert(qkrylov_device_vector_dimension(x_dev32) == 4);
    assert(qkrylov_device_vector_precision(x_dev32) == 0);
    assert(qkrylov_device_vector_data(x_dev32) != nullptr);

    // 3. Staging Copies: Host -> Device -> Host (FP64)
    std::vector<std::complex<double>> host_in64 = {
        {1.0, 0.5}, {-2.0, 1.5}, {0.0, -1.0}, {3.0, 2.0}
    };
    std::vector<std::complex<double>> host_out64(H_dim);
    assert(qkrylov_device_vector_copy_from_host_fp64(x_dev64, reinterpret_cast<const double*>(host_in64.data())) == QKRYLOV_SUCCESS);
    assert(qkrylov_device_vector_copy_to_host_fp64(x_dev64, reinterpret_cast<double*>(host_out64.data())) == QKRYLOV_SUCCESS);
    for (uint64_t i = 0; i < H_dim; ++i) {
        assert(std::abs(host_out64[i] - host_in64[i]) < 1e-15);
    }

    // 4. Staging Copies: Host -> Device -> Host (FP32)
    std::vector<std::complex<float>> host_in32 = {
        {1.0f, 0.5f}, {-2.0f, 1.5f}, {0.0f, -1.0f}, {3.0f, 2.0f}
    };
    std::vector<std::complex<float>> host_out32(H_dim);
    assert(qkrylov_device_vector_copy_from_host_fp32(x_dev32, reinterpret_cast<const float*>(host_in32.data())) == QKRYLOV_SUCCESS);
    assert(qkrylov_device_vector_copy_to_host_fp32(x_dev32, reinterpret_cast<float*>(host_out32.data())) == QKRYLOV_SUCCESS);
    for (uint64_t i = 0; i < H_dim; ++i) {
        assert(std::abs(host_out32[i] - host_in32[i]) < 1e-6f);
    }

    // 5. Zero-Copy SpMV Device Execution: y_dev = H * x_dev (FP64)
    assert(qkrylov_hamiltonian_apply_device_fp64(H_dev64, x_dev64, y_dev64) == QKRYLOV_SUCCESS);
    std::vector<std::complex<double>> y_res_dev64(H_dim);
    assert(qkrylov_device_vector_copy_to_host_fp64(y_dev64, reinterpret_cast<double*>(y_res_dev64.data())) == QKRYLOV_SUCCESS);

    // Compare with CPU host apply
    std::vector<std::complex<double>> y_ref_host64(H_dim);
    assert(qkrylov_hamiltonian_apply_complex_fp64(H_dev64,
        reinterpret_cast<const double*>(host_in64.data()),
        reinterpret_cast<double*>(y_ref_host64.data())) == QKRYLOV_SUCCESS);
    for (uint64_t i = 0; i < H_dim; ++i) {
        assert(std::abs(y_res_dev64[i] - y_ref_host64[i]) < 1e-14);
    }

    // 6. Zero-Copy SpMV Device Execution: y_dev = H * x_dev (FP32)
    assert(qkrylov_hamiltonian_apply_device_fp32(H_dev32, x_dev32, y_dev32) == QKRYLOV_SUCCESS);
    std::vector<std::complex<float>> y_res_dev32(H_dim);
    assert(qkrylov_device_vector_copy_to_host_fp32(y_dev32, reinterpret_cast<float*>(y_res_dev32.data())) == QKRYLOV_SUCCESS);

    std::vector<std::complex<float>> y_ref_host32(H_dim);
    assert(qkrylov_hamiltonian_apply_complex_fp32(H_dev32,
        reinterpret_cast<const float*>(host_in32.data()),
        reinterpret_cast<float*>(y_ref_host32.data())) == QKRYLOV_SUCCESS);
    for (uint64_t i = 0; i < H_dim; ++i) {
        assert(std::abs(y_res_dev32[i] - y_ref_host32[i]) < 1e-5f);
    }

    // 7. Device Diagonal: qkrylov_hamiltonian_diagonal_device
    qkrylov_device_vector_h diag_dev64 = qkrylov_device_vector_create_fp64(H_dim);
    assert(qkrylov_hamiltonian_diagonal_device_fp64(H_dev64, diag_dev64) == QKRYLOV_SUCCESS);
    std::vector<std::complex<double>> diag_host64(H_dim);
    assert(qkrylov_device_vector_copy_to_host_fp64(diag_dev64, reinterpret_cast<double*>(diag_host64.data())) == QKRYLOV_SUCCESS);
    std::vector<double> diag_ref64(H_dim);
    assert(qkrylov_hamiltonian_diagonal_fp64(H_dev64, diag_ref64.data()) == QKRYLOV_SUCCESS);
    for (uint64_t i = 0; i < H_dim; ++i) {
        assert(std::abs(diag_host64[i].real() - diag_ref64[i]) < 1e-14);
    }
    qkrylov_device_vector_destroy(diag_dev64);

    // 8. Device Vector BLAS-1 Kernels (dot, norm, axpy, scal, normalize, zero_fill, copy)
    double dev_dot_re = 0.0, dev_dot_im = 0.0;
    assert(qkrylov_device_vector_dot_fp64(x_dev64, y_dev64, &dev_dot_re, &dev_dot_im) == QKRYLOV_SUCCESS);
    double host_dot_re = 0.0, host_dot_im = 0.0;
    assert(qkrylov_vector_dot_fp64(H_dim,
        reinterpret_cast<const double*>(host_in64.data()),
        reinterpret_cast<const double*>(y_ref_host64.data()),
        &host_dot_re, &host_dot_im) == QKRYLOV_SUCCESS);
    assert(std::abs(dev_dot_re - host_dot_re) < 1e-13);
    assert(std::abs(dev_dot_im - host_dot_im) < 1e-13);

    double dev_nrm = 0.0, host_nrm = 0.0;
    assert(qkrylov_device_vector_norm_fp64(x_dev64, &dev_nrm) == QKRYLOV_SUCCESS);
    assert(qkrylov_vector_norm_fp64(H_dim, reinterpret_cast<const double*>(host_in64.data()), &host_nrm) == QKRYLOV_SUCCESS);
    assert(std::abs(dev_nrm - host_nrm) < 1e-13);

    assert(qkrylov_device_vector_axpy_fp64(2.0, -1.0, x_dev64, y_dev64) == QKRYLOV_SUCCESS);
    assert(qkrylov_device_vector_scal_fp64(0.5, 0.0, y_dev64) == QKRYLOV_SUCCESS);
    assert(qkrylov_device_vector_normalize_fp64(y_dev64) == QKRYLOV_SUCCESS);
    double dev_unit_nrm = 0.0;
    assert(qkrylov_device_vector_norm_fp64(y_dev64, &dev_unit_nrm) == QKRYLOV_SUCCESS);
    assert(std::abs(dev_unit_nrm - 1.0) < 1e-13);

    qkrylov_device_vector_h clone64 = qkrylov_device_vector_create_fp64(H_dim);
    assert(qkrylov_device_vector_copy_fp64(y_dev64, clone64) == QKRYLOV_SUCCESS);
    double clone_nrm = 0.0;
    assert(qkrylov_device_vector_norm_fp64(clone64, &clone_nrm) == QKRYLOV_SUCCESS);
    assert(std::abs(clone_nrm - 1.0) < 1e-13);

    assert(qkrylov_device_vector_zero_fill_fp64(clone64) == QKRYLOV_SUCCESS);
    assert(qkrylov_device_vector_norm_fp64(clone64, &clone_nrm) == QKRYLOV_SUCCESS);
    assert(clone_nrm == 0.0);

    // 9. Error diagnostics
    assert(qkrylov_hamiltonian_apply_device_fp64(H_dev64, x_dev32, y_dev64) == QKRYLOV_ERROR_INVALID_ARG); // Precision mismatch
    assert(std::string(qkrylov_get_last_error_message()).find("precision mismatch") != std::string::npos);
    qkrylov_clear_last_error();

    qkrylov_device_vector_h mismatch_dim = qkrylov_device_vector_create_fp64(8);
    assert(qkrylov_hamiltonian_apply_device_fp64(H_dev64, mismatch_dim, y_dev64) == QKRYLOV_ERROR_INVALID_ARG); // Dim mismatch
    assert(std::string(qkrylov_get_last_error_message()).find("dimension") != std::string::npos);
    qkrylov_clear_last_error();

    assert(qkrylov_device_vector_copy_from_host_fp64(nullptr, nullptr) == QKRYLOV_ERROR_INVALID_ARG);
    qkrylov_clear_last_error();

    // 10. Cleanup
    qkrylov_device_vector_destroy(x_dev64);
    qkrylov_device_vector_destroy(y_dev64);
    qkrylov_device_vector_destroy(x_dev32);
    qkrylov_device_vector_destroy(y_dev32);
    qkrylov_device_vector_destroy(clone64);
    qkrylov_device_vector_destroy(mismatch_dim);
    qkrylov_hamiltonian_destroy(H_dev64);
    qkrylov_hamiltonian_destroy(H_dev32);
    qkrylov_opsum_destroy(ops_dev);
    qkrylov_site_destroy(s_dev);
    qkrylov_basis_destroy(b_dev);

    std::cout << "Device-Resident Vectors & Zero-Copy SpMV verified successfully." << std::endl;

    std::cout << "\nAll Dual-Precision C API tests passed successfully!" << std::endl;
    return 0;
}
