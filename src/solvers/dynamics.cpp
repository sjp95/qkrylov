#include "qkrylov/core/types.hpp"
#include "qkrylov/solvers/dynamics.hpp"
#include "qkrylov/solvers/lanczos.hpp"
#include "qkrylov/linalg/tridiag_qr.hpp"
#include "qkrylov/linalg/vector_ops.hpp"

#include <cmath>
#include <complex>
#include <limits>
#include <random>
#include <algorithm>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {

template <typename ExecSpace>
DynamicsResult continued_fraction_coeffs(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    const HostVector& phi0,
    int n_iter
)
{
    const Index dim = H.dimension();
    if (dim == 0 || phi0.empty()) return {};

    Real norm_sq = Real(0.0);
    for (Index i = 0; i < phi0.size(); ++i) {
        norm_sq += std::norm(phi0[i]);
    }
    const Real norm_phi = std::sqrt(norm_sq);

    const Real mach_eps = std::numeric_limits<Real>::epsilon() * Real(4.0);
    if (norm_phi < mach_eps) return { {}, {}, Real(0.0) };

    LanczosConfig cfg;
    cfg.maxiter = n_iter;
    cfg.min_iterations = n_iter;
    cfg.tol = Real(0.0);
    cfg.initial_vector = phi0;
    cfg.compute_eigenvectors = false;

    auto l_res = solvers::lanczos<solvers::policy::OnePass>(H, cfg);

    DynamicsResult res;
    res.alphas = std::move(l_res.alphas);
    res.betas = std::move(l_res.betas);
    res.norm_phi0 = norm_phi;
    return res;
}

Real evaluate_spectral_function(
    const Real* alphas,
    const Real* betas,
    size_t n,
    Real norm_phi0,
    Real omega,
    Real E0,
    Real eta
)
{
    if (n == 0) return 0.0;

    std::complex<Real> z(omega + E0, eta);

    // Backward recursion for continued fraction
    // C_n = 1 / (z - a_n)
    // C_{i} = 1 / (z - a_i - b_i^2 * C_{i+1})

    std::complex<Real> f = 0.0;
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        if (i == static_cast<int>(n) - 1) {
            f = Real(1.0) / (z - alphas[i]);
        } else {
            f = Real(1.0) / (z - alphas[i] - betas[i] * betas[i] * f);
        }
    }

    constexpr Real PI = static_cast<Real>(3.141592653589793238462643383279502884L);
    return -static_cast<Real>(1.0) / PI * std::imag(norm_phi0 * norm_phi0 * f);
}

template <typename ExecSpace>
RealTimeResult time_evolve(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    const HostVector& psi0,
    const std::vector<Real>& time_grid,
    const std::vector<MatrixFreeHamiltonian<ExecSpace>>& observables,
    int n_steps
)
{
    const Index dim = H.dimension();
    if (dim == 0 || psi0.empty() || time_grid.empty()) return {};

    Real norm_sq = Real(0.0);
    for (Index i = 0; i < psi0.size(); ++i) {
        norm_sq += std::norm(psi0[i]);
    }
    const Real norm_psi = std::sqrt(norm_sq);
    const Real mach_eps = std::numeric_limits<Real>::epsilon() * Real(4.0);
    if (norm_psi < mach_eps) return {};

    const int max_steps = std::min<int>(n_steps, static_cast<int>(dim));

    VectorView<ExecSpace> v_prev("v_prev", dim);
    VectorView<ExecSpace> v_curr("v_curr", dim);
    VectorView<ExecSpace> w("w", dim);

    copy_host_to_device(psi0, v_curr);
    scal(Real(1.0) / norm_psi, v_curr);
    zero_fill(v_prev);

    std::vector<VectorView<ExecSpace>> basis_vectors;
    basis_vectors.reserve(max_steps);

    std::vector<Real> alphas;
    std::vector<Real> betas;
    alphas.reserve(max_steps);
    betas.reserve(max_steps);

    for (int j = 0; j < max_steps; ++j) {
        VectorView<ExecSpace> v_saved("basis_v", dim);
        Kokkos::deep_copy(v_saved, v_curr);
        basis_vectors.push_back(v_saved);

        H.apply(v_curr, w);
        Real alpha = dot(v_curr, w).real();
        alphas.push_back(alpha);

        axpy(KComplex(-alpha, 0.0), v_curr, w);
        if (j > 0) {
            axpy(KComplex(-betas.back(), 0.0), v_prev, w);
        }

        // Full reorthogonalization to preserve exact unitary norms
        for (int k = 0; k <= j; ++k) {
            KComplex ovlp = dot(basis_vectors[k], w);
            axpy(-ovlp, basis_vectors[k], w);
        }

        Real beta_val = norm(w);
        if (beta_val < mach_eps) break;
        if (j + 1 == max_steps) break;
        betas.push_back(beta_val);

        Kokkos::deep_copy(v_prev, v_curr);
        Kokkos::deep_copy(v_curr, w);
        scal(Real(1.0) / beta_val, v_curr);
    }

    const int M = static_cast<int>(alphas.size());
    auto eig = linalg::tridiag_eigensystem_full(alphas, betas, M);

    const size_t num_times = time_grid.size();
    const size_t num_obs = observables.size();

    // Project observables onto Krylov basis: O_jk = <v_j | O | v_k>
    std::vector<std::vector<Complex>> projected_obs(num_obs);
    for (size_t oi = 0; oi < num_obs; ++oi) {
        projected_obs[oi].resize(M * M);
        for (int k = 0; k < M; ++k) {
            observables[oi].apply(basis_vectors[k], w);
            for (int j = 0; j < M; ++j) {
                KComplex val = dot(basis_vectors[j], w);
                projected_obs[oi][j * M + k] = Complex(val.real(), val.imag());
            }
        }
    }

    RealTimeResult res;
    res.time_grid = time_grid;
    res.survival_probabilities.resize(num_times);
    res.observable_expectations.resize(num_obs, std::vector<Complex>(num_times, Complex(0.0, 0.0)));

    for (size_t ti = 0; ti < num_times; ++ti) {
        Real t = time_grid[ti];

        std::vector<Complex> c(M, Complex(0.0, 0.0));
        Complex overlap(0.0, 0.0);

        for (int m = 0; m < M; ++m) {
            Real theta = -eig.eigenvalues[m] * t;
            Complex phase(std::cos(theta), std::sin(theta));
            Real v0 = eig.eigenvectors[m][0];
            overlap += (v0 * v0) * phase;

            Complex factor = v0 * phase;
            for (int j = 0; j < M; ++j) {
                c[j] += factor * eig.eigenvectors[m][j];
            }
        }

        res.survival_probabilities[ti] = overlap;

        for (size_t oi = 0; oi < num_obs; ++oi) {
            const auto& mat = projected_obs[oi];
            Complex val(0.0, 0.0);
            for (int j = 0; j < M; ++j) {
                for (int k = 0; k < M; ++k) {
                    val += std::conj(c[j]) * mat[j * M + k] * c[k];
                }
            }
            res.observable_expectations[oi][ti] = val;
        }
    }

    return res;
}

template <typename ExecSpace>
FTLMDynamicsResult ftlm_dynamics(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    Real beta,
    const MatrixFreeHamiltonian<ExecSpace>& A,
    const MatrixFreeHamiltonian<ExecSpace>& B,
    const std::vector<Real>& time_grid,
    int n_random,
    int n_steps,
    uint64_t seed
)
{
    const Index dim = H.dimension();
    if (dim == 0 || n_random <= 0 || n_steps <= 0 || time_grid.empty()) return {};

    const size_t R = static_cast<size_t>(n_random);
    const size_t num_times = time_grid.size();
    const int max_steps = std::min<int>(n_steps, static_cast<int>(dim));
    const Real mach_eps = std::numeric_limits<Real>::epsilon() * Real(4.0);

    // Pass 1: Find global ground-state energy shift E_min
    Real E_min = std::numeric_limits<Real>::infinity();
    {
        std::mt19937_64 rng(seed);
        std::normal_distribution<Real> dist(Real(0.0), Real(1.0));

        VectorView<ExecSpace> v_prev("v_prev", dim);
        VectorView<ExecSpace> v_curr("v_curr", dim);
        VectorView<ExecSpace> w("w", dim);

        for (size_t r = 0; r < R; ++r) {
            HostVector r_vec(dim);
            Real sum_sq = Real(0.0);
            for (Index i = 0; i < dim; ++i) {
                Complex c(dist(rng), dist(rng));
                r_vec[i] = c;
                sum_sq += std::norm(c);
            }
            Real nrm = std::sqrt(sum_sq);
            if (nrm < mach_eps) nrm = Real(1.0);

            copy_host_to_device(r_vec, v_curr);
            scal(Real(1.0) / nrm, v_curr);
            zero_fill(v_prev);

            std::vector<Real> alphas;
            std::vector<Real> betas;
            for (int j = 0; j < max_steps; ++j) {
                H.apply(v_curr, w);
                Real alpha = dot(v_curr, w).real();
                alphas.push_back(alpha);

                axpy(KComplex(-alpha, 0.0), v_curr, w);
                if (j > 0) axpy(KComplex(-betas.back(), 0.0), v_prev, w);

                Real beta_val = norm(w);
                if (beta_val < mach_eps) break;
                if (j + 1 == max_steps) break;
                betas.push_back(beta_val);

                Kokkos::deep_copy(v_prev, v_curr);
                Kokkos::deep_copy(v_curr, w);
                scal(Real(1.0) / beta_val, v_curr);
            }

            int M = static_cast<int>(alphas.size());
            auto eig = linalg::tridiag_eigensystem_full(alphas, betas, M);
            if (!eig.eigenvalues.empty() && eig.eigenvalues[0] < E_min) {
                E_min = eig.eigenvalues[0];
            }
        }
    }

    if (std::isinf(E_min)) return {};

    // Pass 2: Streamed real-time correlator evaluation
    Real Z_sum = Real(0.0);
    std::vector<Complex> C_sum(num_times, Complex(0.0, 0.0));
    std::vector<Real> sample_Z(R, Real(0.0));
    std::vector<std::vector<Complex>> sample_C(R, std::vector<Complex>(num_times, Complex(0.0, 0.0)));

    {
        std::mt19937_64 rng(seed);
        std::normal_distribution<Real> dist(Real(0.0), Real(1.0));

        VectorView<ExecSpace> v_prev("v_prev", dim);
        VectorView<ExecSpace> v_curr("v_curr", dim);
        VectorView<ExecSpace> w("w", dim);

        for (size_t r = 0; r < R; ++r) {
            HostVector r_vec(dim);
            Real sum_sq = Real(0.0);
            for (Index i = 0; i < dim; ++i) {
                Complex c(dist(rng), dist(rng));
                r_vec[i] = c;
                sum_sq += std::norm(c);
            }
            Real nrm = std::sqrt(sum_sq);
            if (nrm < mach_eps) nrm = Real(1.0);

            copy_host_to_device(r_vec, v_curr);
            scal(Real(1.0) / nrm, v_curr);
            zero_fill(v_prev);

            std::vector<VectorView<ExecSpace>> basis_vectors;
            basis_vectors.reserve(max_steps);
            std::vector<Real> alphas;
            std::vector<Real> betas;

            for (int j = 0; j < max_steps; ++j) {
                VectorView<ExecSpace> v_saved("basis_v", dim);
                Kokkos::deep_copy(v_saved, v_curr);
                basis_vectors.push_back(v_saved);

                H.apply(v_curr, w);
                Real alpha = dot(v_curr, w).real();
                alphas.push_back(alpha);

                axpy(KComplex(-alpha, 0.0), v_curr, w);
                if (j > 0) axpy(KComplex(-betas.back(), 0.0), v_prev, w);

                Real beta_val = norm(w);
                if (beta_val < mach_eps) break;
                if (j + 1 == max_steps) break;
                betas.push_back(beta_val);

                Kokkos::deep_copy(v_prev, v_curr);
                Kokkos::deep_copy(v_curr, w);
                scal(Real(1.0) / beta_val, v_curr);
            }

            const int M = static_cast<int>(alphas.size());
            auto eig = linalg::tridiag_eigensystem_full(alphas, betas, M);

            // Thermal state |phi_r(beta/2)> components and sample partition weight
            Real z_r = Real(0.0);
            std::vector<Real> c_phi(M, Real(0.0));
            for (int m = 0; m < M; ++m) {
                Real exp_arg = -beta * (eig.eigenvalues[m] - E_min);
                Real exp_val = (exp_arg < -Real(80.0)) ? Real(0.0) : std::exp(exp_arg);
                Real v0 = eig.eigenvectors[m][0];
                z_r += v0 * v0 * exp_val;

                Real half_exp_arg = -beta * (eig.eigenvalues[m] - E_min) * Real(0.5);
                Real half_exp_val = (half_exp_arg < -Real(80.0)) ? Real(0.0) : std::exp(half_exp_arg);
                Real factor = v0 * half_exp_val;
                for (int j = 0; j < M; ++j) {
                    c_phi[j] += factor * eig.eigenvectors[m][j];
                }
            }
            sample_Z[r] = z_r;
            Z_sum += z_r;

            // Form thermal state |phi> in full Hilbert space
            VectorView<ExecSpace> phi_vec("phi_vec", dim);
            zero_fill(phi_vec);
            for (int j = 0; j < M; ++j) {
                axpy(KComplex(c_phi[j], 0.0), basis_vectors[j], phi_vec);
            }

            // Apply B to |phi>: |chi_0> = B |phi>
            VectorView<ExecSpace> chi_0("chi_0", dim);
            B.apply(phi_vec, chi_0);
            Real chi_norm = norm(chi_0);

            if (chi_norm < mach_eps) {
                for (size_t ti = 0; ti < num_times; ++ti) {
                    sample_C[r][ti] = Complex(0.0, 0.0);
                }
            } else {
                // Run Lanczos for |chi_0>
                VectorView<ExecSpace> u_prev("u_prev", dim);
                VectorView<ExecSpace> u_curr("u_curr", dim);
                VectorView<ExecSpace> u_w("u_w", dim);

                Kokkos::deep_copy(u_curr, chi_0);
                scal(Real(1.0) / chi_norm, u_curr);
                zero_fill(u_prev);

                std::vector<VectorView<ExecSpace>> chi_basis;
                chi_basis.reserve(max_steps);
                std::vector<Real> chi_alphas;
                std::vector<Real> chi_betas;

                for (int j = 0; j < max_steps; ++j) {
                    VectorView<ExecSpace> u_saved("chi_basis_v", dim);
                    Kokkos::deep_copy(u_saved, u_curr);
                    chi_basis.push_back(u_saved);

                    H.apply(u_curr, u_w);
                    Real alpha = dot(u_curr, u_w).real();
                    chi_alphas.push_back(alpha);

                    axpy(KComplex(-alpha, 0.0), u_curr, u_w);
                    if (j > 0) axpy(KComplex(-chi_betas.back(), 0.0), u_prev, u_w);

                    Real b_val = norm(u_w);
                    if (b_val < mach_eps) break;
                    if (j + 1 == max_steps) break;
                    chi_betas.push_back(b_val);

                    Kokkos::deep_copy(u_prev, u_curr);
                    Kokkos::deep_copy(u_curr, u_w);
                    scal(Real(1.0) / b_val, u_curr);
                }

                const int M_chi = static_cast<int>(chi_alphas.size());
                auto eig_chi = linalg::tridiag_eigensystem_full(chi_alphas, chi_betas, M_chi);

                // Project operator A between the two Krylov bases: A_jk = <v_j | A | w_k>
                std::vector<Complex> mat_A_cross(M * M_chi);
                for (int k = 0; k < M_chi; ++k) {
                    A.apply(chi_basis[k], w);
                    for (int j = 0; j < M; ++j) {
                        KComplex val = dot(basis_vectors[j], w);
                        mat_A_cross[j * M_chi + k] = Complex(val.real(), val.imag());
                    }
                }

                // Propagate in real time across time_grid
                for (size_t ti = 0; ti < num_times; ++ti) {
                    Real t = time_grid[ti];

                    // Real-time evolved |phi(t)> coefficients in basis_vectors
                    std::vector<Complex> phi_t(M, Complex(0.0, 0.0));
                    for (int m = 0; m < M; ++m) {
                        Real theta = -eig.eigenvalues[m] * t;
                        Complex phase(std::cos(theta), std::sin(theta));

                        Real half_exp_arg = -beta * (eig.eigenvalues[m] - E_min) * Real(0.5);
                        Real half_exp_val = (half_exp_arg < -Real(80.0)) ? Real(0.0) : std::exp(half_exp_arg);
                        Complex factor_phi = (eig.eigenvectors[m][0] * half_exp_val) * phase;

                        for (int j = 0; j < M; ++j) {
                            phi_t[j] += factor_phi * eig.eigenvectors[m][j];
                        }
                    }

                    // Real-time evolved |chi(t)> coefficients in chi_basis
                    std::vector<Complex> chi_t(M_chi, Complex(0.0, 0.0));
                    for (int m = 0; m < M_chi; ++m) {
                        Real theta = -eig_chi.eigenvalues[m] * t;
                        Complex phase(std::cos(theta), std::sin(theta));
                        Complex factor_chi = eig_chi.eigenvectors[m][0] * phase;

                        for (int k = 0; k < M_chi; ++k) {
                            chi_t[k] += factor_chi * eig_chi.eigenvectors[m][k];
                        }
                    }

                    // Contract: <phi(t) | A | chi(t)>
                    Complex corr(0.0, 0.0);
                    for (int j = 0; j < M; ++j) {
                        for (int k = 0; k < M_chi; ++k) {
                            corr += std::conj(phi_t[j]) * mat_A_cross[j * M_chi + k] * chi_t[k];
                        }
                    }
                    corr *= chi_norm;

                    sample_C[r][ti] = corr;
                    C_sum[ti] += corr;
                }
            }

            basis_vectors.clear();
        }
    }

    FTLMDynamicsResult res;
    res.beta = beta;
    res.time_grid = time_grid;
    res.correlations.resize(num_times, Complex(0.0, 0.0));
    res.correlation_errors.resize(num_times, Real(0.0));

    Real Z_bar = Z_sum / Real(R);
    if (Z_sum > Real(0.0) && Z_bar > Real(0.0)) {
        for (size_t ti = 0; ti < num_times; ++ti) {
            Complex mean_c = C_sum[ti] / Z_sum;
            res.correlations[ti] = mean_c;

            if (R > 1) {
                Real var_linearized = Real(0.0);
                for (size_t r = 0; r < R; ++r) {
                    Complex delta_r = sample_C[r][ti] - mean_c * sample_Z[r];
                    var_linearized += std::norm(delta_r);
                }
                Real denom = Real(R) * Real(R - 1) * (Z_bar * Z_bar);
                if (denom > Real(0.0)) {
                    res.correlation_errors[ti] = std::sqrt(var_linearized / denom);
                }
            }
        }
    }

    return res;
}

// Explicit instantiations
#define INSTANTIATE_DYNAMICS(EXEC_SPACE) \
template DynamicsResult continued_fraction_coeffs<EXEC_SPACE>(const MatrixFreeHamiltonian<EXEC_SPACE>&, const HostVector&, int); \
template RealTimeResult time_evolve<EXEC_SPACE>(const MatrixFreeHamiltonian<EXEC_SPACE>&, const HostVector&, const std::vector<Real>&, const std::vector<MatrixFreeHamiltonian<EXEC_SPACE>>&, int); \
template FTLMDynamicsResult ftlm_dynamics<EXEC_SPACE>(const MatrixFreeHamiltonian<EXEC_SPACE>&, Real, const MatrixFreeHamiltonian<EXEC_SPACE>&, const MatrixFreeHamiltonian<EXEC_SPACE>&, const std::vector<Real>&, int, int, uint64_t);

#ifdef KOKKOS_ENABLE_SERIAL
INSTANTIATE_DYNAMICS(Kokkos::Serial)
#endif
#ifdef KOKKOS_ENABLE_OPENMP
INSTANTIATE_DYNAMICS(Kokkos::OpenMP)
#endif
#ifdef KOKKOS_ENABLE_THREADS
INSTANTIATE_DYNAMICS(Kokkos::Threads)
#endif
#ifdef KOKKOS_ENABLE_CUDA
INSTANTIATE_DYNAMICS(Kokkos::Cuda)
#endif
#ifdef KOKKOS_ENABLE_HIP
INSTANTIATE_DYNAMICS(Kokkos::HIP)
#endif
#ifdef KOKKOS_ENABLE_SYCL
INSTANTIATE_DYNAMICS(Kokkos::Experimental::SYCL)
#endif

#undef INSTANTIATE_DYNAMICS

} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov
