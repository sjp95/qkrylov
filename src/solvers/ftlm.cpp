#include "qkrylov/core/types.hpp"
#include "qkrylov/solvers/ftlm.hpp"
#include "qkrylov/solvers/policy.hpp"
#include "qkrylov/linalg/tridiag_qr.hpp"
#include "qkrylov/linalg/vector_ops.hpp"

#include <random>
#include <cmath>
#include <algorithm>
#include <limits>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {

template <typename ExecSpace, typename Policy>
std::vector<FTLMKrylovSample> ftlm_sample(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    const std::vector<MatrixFreeHamiltonian<ExecSpace>>& observables,
    int n_random,
    int n_steps,
    uint64_t seed
)
{
    const Index dim = H.dimension();
    if (dim == 0 || n_random <= 0 || n_steps <= 0) return {};

    std::mt19937_64 rng(seed);
    std::normal_distribution<Real> dist(Real(0.0), Real(1.0));

    const Real mach_eps = std::numeric_limits<Real>::epsilon() * Real(4.0);
    const int max_steps = std::min<int>(n_steps, static_cast<int>(dim));

    std::vector<FTLMKrylovSample> samples(n_random);

    for (int r = 0; r < n_random; ++r) {
        HostVector r_vec_host(dim);
        Real sum_sq = Real(0.0);
        for (Index i = 0; i < dim; ++i) {
            Complex c(dist(rng), dist(rng));
            r_vec_host[i] = c;
            sum_sq += std::norm(c);
        }
        Real nrm = std::sqrt(sum_sq);
        if (nrm < mach_eps) nrm = Real(1.0);

        VectorView<ExecSpace> v_prev("v_prev", dim);
        VectorView<ExecSpace> v_curr("v_curr", dim);
        VectorView<ExecSpace> w("w", dim);

        copy_host_to_device(r_vec_host, v_curr);
        scal(Real(1.0) / nrm, v_curr);
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

            Real beta_val = norm(w);
            if (beta_val < mach_eps) {
                break;
            }
            if (j + 1 == max_steps) {
                break;
            }
            betas.push_back(beta_val);

            Kokkos::deep_copy(v_prev, v_curr);
            Kokkos::deep_copy(v_curr, w);
            scal(Real(1.0) / beta_val, v_curr);
        }

        const int M = static_cast<int>(alphas.size());
        auto eig = linalg::tridiag_eigensystem_full(alphas, betas, M);

        samples[r].norm = nrm;
        samples[r].eigenvalues = eig.eigenvalues;
        samples[r].first_components.resize(M);
        for (int m = 0; m < M; ++m) {
            samples[r].first_components[m] = eig.eigenvectors[m][0];
        }
        samples[r].eigenvectors = std::move(eig.eigenvectors);

        // Project observables onto the Krylov subspace: O_jk = <v_j | O | v_k>
        if (!observables.empty()) {
            samples[r].projected_operators.resize(observables.size());
            for (size_t oi = 0; oi < observables.size(); ++oi) {
                const auto& obs = observables[oi];
                ProjectedOperator proj;
                proj.matrix.resize(M * M);

                for (int k = 0; k < M; ++k) {
                    obs.apply(basis_vectors[k], w);
                    for (int j = 0; j < M; ++j) {
                        KComplex val = dot(basis_vectors[j], w);
                        proj.matrix[j * M + k] = Complex(val.real(), val.imag());
                    }
                }
                samples[r].projected_operators[oi] = std::move(proj);
            }
        }
    }

    return samples;
}

FTLMSweepResult ftlm_evaluate_sweep(
    const std::vector<FTLMKrylovSample>& samples,
    const std::vector<Real>& beta_grid
)
{
    if (samples.empty() || beta_grid.empty()) return {};

    const size_t R = samples.size();
    const size_t num_betas = beta_grid.size();

    Real E_min = std::numeric_limits<Real>::infinity();
    size_t num_obs = 0;
    for (const auto& s : samples) {
        if (!s.eigenvalues.empty() && s.eigenvalues[0] < E_min) {
            E_min = s.eigenvalues[0];
        }
        if (s.projected_operators.size() > num_obs) {
            num_obs = s.projected_operators.size();
        }
    }

    if (std::isinf(E_min)) return {};

    FTLMSweepResult res;
    res.beta_grid = beta_grid;
    res.partition_functions.resize(num_betas, Real(0.0));
    res.free_energies.resize(num_betas, Real(0.0));
    res.internal_energies.resize(num_betas, Real(0.0));
    res.specific_heats.resize(num_betas, Real(0.0));
    res.entropies.resize(num_betas, Real(0.0));
    res.observable_expectations.resize(num_obs, std::vector<Real>(num_betas, Real(0.0)));
    res.observable_errors.resize(num_obs, std::vector<Real>(num_betas, Real(0.0)));

    const Real max_exp = (sizeof(Real) > 4) ? Real(700.0) : Real(85.0);

    for (size_t bi = 0; bi < num_betas; ++bi) {
        const Real beta = beta_grid[bi];

        Real Z_shifted = Real(0.0);
        Real E_shifted = Real(0.0);
        Real E2_shifted = Real(0.0);

        std::vector<Real> A_shifted(num_obs, Real(0.0));
        std::vector<std::vector<Real>> sample_obs(num_obs, std::vector<Real>(R, Real(0.0)));

        for (size_t r = 0; r < R; ++r) {
            const auto& s = samples[r];
            const Real nrm = s.norm;
            const int M = static_cast<int>(s.eigenvalues.size());
            if (M == 0) continue;

            Real sample_Z = Real(0.0);
            std::vector<Real> c(M, Real(0.0));

            for (int m = 0; m < M; ++m) {
                Real exp_arg = -beta * (s.eigenvalues[m] - E_min);
                Real exp_val = (exp_arg < -Real(80.0)) ? Real(0.0) : std::exp(exp_arg);
                Real v0 = s.first_components[m];
                Real weight = nrm * nrm * v0 * v0 * exp_val;

                sample_Z += weight;
                Z_shifted += weight;
                E_shifted += s.eigenvalues[m] * weight;
                E2_shifted += s.eigenvalues[m] * s.eigenvalues[m] * weight;

                Real half_exp_arg = -beta * (s.eigenvalues[m] - E_min) * Real(0.5);
                Real half_exp_val = (half_exp_arg < -Real(80.0)) ? Real(0.0) : std::exp(half_exp_arg);
                Real factor = nrm * v0 * half_exp_val;
                for (int j = 0; j < M; ++j) {
                    c[j] += factor * s.eigenvectors[m][j];
                }
            }

            // Project observables for this sample: c^T * Re(O) * c
            for (size_t oi = 0; oi < num_obs; ++oi) {
                if (oi >= s.projected_operators.size()) continue;
                const auto& mat = s.projected_operators[oi].matrix;
                Real obs_val = Real(0.0);
                for (int j = 0; j < M; ++j) {
                    for (int k = 0; k < M; ++k) {
                        obs_val += c[j] * c[k] * mat[j * M + k].real();
                    }
                }
                A_shifted[oi] += obs_val;
                sample_obs[oi][r] = (sample_Z > Real(0.0)) ? (obs_val / sample_Z) : Real(0.0);
            }
        }

        Z_shifted /= Real(R);
        E_shifted /= Real(R);
        E2_shifted /= Real(R);

        if (Z_shifted > Real(0.0)) {
            Real log_Z = std::log(Z_shifted) - beta * E_min;
            res.partition_functions[bi] = (log_Z < max_exp) ? std::exp(log_Z) : std::numeric_limits<Real>::infinity();
            Real mean_E = E_shifted / Z_shifted;
            res.internal_energies[bi] = mean_E;
            res.free_energies[bi] = (beta > Real(0.0)) ? (E_min - std::log(Z_shifted) / beta) : Real(0.0);
            Real var_E = E2_shifted / Z_shifted - mean_E * mean_E;
            if (var_E < Real(0.0)) var_E = Real(0.0);
            res.specific_heats[bi] = (beta * beta) * var_E;
            Real s_val = (beta > Real(0.0)) ? (beta * (mean_E - res.free_energies[bi])) : std::log(Z_shifted);
            if (s_val < Real(0.0)) s_val = Real(0.0);
            res.entropies[bi] = s_val;

            for (size_t oi = 0; oi < num_obs; ++oi) {
                Real mean_obs = (A_shifted[oi] / Real(R)) / Z_shifted;
                res.observable_expectations[oi][bi] = mean_obs;

                if (R > 1) {
                    Real var = Real(0.0);
                    for (size_t r = 0; r < R; ++r) {
                        Real diff = sample_obs[oi][r] - mean_obs;
                        var += diff * diff;
                    }
                    var /= Real(R - 1);
                    res.observable_errors[oi][bi] = std::sqrt(var / Real(R));
                } else {
                    res.observable_errors[oi][bi] = Real(0.0);
                }
            }
        }
    }

    return res;
}

template <typename ExecSpace, typename Policy>
FTLMSweepResult ftlm_sweep(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    const std::vector<Real>& beta_grid,
    const std::vector<MatrixFreeHamiltonian<ExecSpace>>& observables,
    int n_random,
    int n_steps,
    uint64_t seed
)
{
    auto samples = ftlm_sample<ExecSpace, Policy>(H, observables, n_random, n_steps, seed);
    return ftlm_evaluate_sweep(samples, beta_grid);
}

template <typename ExecSpace, typename Policy>
FTLMResult ftlm(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    Real beta,
    int n_random,
    int n_steps,
    uint64_t seed
)
{
    auto sweep = ftlm_sweep<ExecSpace, Policy>(H, {beta}, {}, n_random, n_steps, seed);
    FTLMResult res;
    res.beta = beta;
    if (!sweep.partition_functions.empty()) {
        res.partition_function = sweep.partition_functions[0];
        res.free_energy = sweep.free_energies[0];
        res.internal_energy = sweep.internal_energies[0];
        res.specific_heat = sweep.specific_heats[0];
        res.entropy = sweep.entropies[0];
    }
    return res;
}

// Explicit instantiations
#define INSTANTIATE_FTLM(EXEC_SPACE) \
template std::vector<FTLMKrylovSample> ftlm_sample<EXEC_SPACE, solvers::policy::OnePass_full>( \
    const MatrixFreeHamiltonian<EXEC_SPACE>&, \
    const std::vector<MatrixFreeHamiltonian<EXEC_SPACE>>&, \
    int, int, uint64_t); \
template FTLMSweepResult ftlm_sweep<EXEC_SPACE, solvers::policy::OnePass_full>( \
    const MatrixFreeHamiltonian<EXEC_SPACE>&, \
    const std::vector<Real>&, \
    const std::vector<MatrixFreeHamiltonian<EXEC_SPACE>>&, \
    int, int, uint64_t); \
template FTLMResult ftlm<EXEC_SPACE, solvers::policy::OnePass_full>( \
    const MatrixFreeHamiltonian<EXEC_SPACE>&, \
    Real, int, int, uint64_t);

#ifdef KOKKOS_ENABLE_SERIAL
INSTANTIATE_FTLM(Kokkos::Serial)
#endif
#ifdef KOKKOS_ENABLE_OPENMP
INSTANTIATE_FTLM(Kokkos::OpenMP)
#endif
#ifdef KOKKOS_ENABLE_THREADS
INSTANTIATE_FTLM(Kokkos::Threads)
#endif
#ifdef KOKKOS_ENABLE_CUDA
INSTANTIATE_FTLM(Kokkos::Cuda)
#endif
#ifdef KOKKOS_ENABLE_HIP
INSTANTIATE_FTLM(Kokkos::HIP)
#endif
#ifdef KOKKOS_ENABLE_SYCL
INSTANTIATE_FTLM(Kokkos::Experimental::SYCL)
#endif

#undef INSTANTIATE_FTLM

} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov
