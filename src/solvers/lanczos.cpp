#include "qkrylov/core/types.hpp"
#include "qkrylov/solvers/lanczos.hpp"
#include "qkrylov/linalg/tridiag_qr.hpp"

#include <random>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <Kokkos_Core.hpp>

namespace qkrylov {
namespace solvers {

template <typename Policy, typename ExecSpace>
QKRYLOV_PRECISION_NAMESPACE::LanczosResult lanczos(
    const QKRYLOV_PRECISION_NAMESPACE::MatrixFreeHamiltonian<ExecSpace>& H,
    const LanczosConfig& config
)
{
    using namespace QKRYLOV_PRECISION_NAMESPACE;
    using Traits = policy::policy_traits<Policy>;
    static_assert(policy::is_policy_v<Policy>, "Unknown or invalid solver policy");

    const Index dim = H.dimension();
    if (dim == 0) return {};

    // 1. Policy Scope Enforcement
    if constexpr (!Traits::supports_multistate) {
        if (config.n_eig > 1) {
            throw std::invalid_argument(
                "Selected Lanczos policy only supports ground-state calculations (n_eig = 1). "
                "Use policy::OnePass_DKGS for an arbitrary number of low energy states."
            );
        }
    }

    const int target_n_eig = Traits::supports_multistate
        ? std::clamp(config.n_eig, 1, static_cast<int>(dim))
        : 1;

    // 2. Working Triad (Strictly O(N))
    VectorView<ExecSpace> v_prev("v_prev", dim);
    VectorView<ExecSpace> v_curr("v_curr", dim);
    VectorView<ExecSpace> w("w", dim);

    std::vector<VectorView<ExecSpace>> basis_vectors;
    if constexpr (Traits::stores_basis) {
        basis_vectors.reserve(config.maxiter);
    }

    // 3. State Initialization
    const Real mach_eps = std::numeric_limits<Real>::epsilon() * Real(4.0);
    const uint64_t seed = config.seed.value_or(123456789ULL);
    auto v_curr_host = Kokkos::create_mirror_view(v_curr);

    if (!config.initial_vector.empty()) {
        if (static_cast<Index>(config.initial_vector.size()) != dim) {
            throw std::invalid_argument("initial_vector size (" + std::to_string(config.initial_vector.size()) +
                                       ") does not match Hamiltonian dimension (" + std::to_string(dim) + ")");
        }
        for (Index i = 0; i < dim; ++i) {
            v_curr_host(i) = KComplex(config.initial_vector[i].real(), config.initial_vector[i].imag());
        }
        Kokkos::deep_copy(v_curr, v_curr_host);
        Real init_norm = norm(v_curr);
        if (init_norm < mach_eps) {
            throw std::invalid_argument("initial_vector has zero norm");
        }
        normalize(v_curr);
    } else {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<Real> dist(-1.0, 1.0);
        for (Index i = 0; i < dim; ++i) {
            v_curr_host(i) = KComplex(dist(rng), dist(rng));
        }
        Kokkos::deep_copy(v_curr, v_curr_host);
        normalize(v_curr);
    }
    zero_fill(v_prev);

    std::vector<Real> alphas;
    std::vector<Real> betas;
    alphas.reserve(config.maxiter);
    betas.reserve(config.maxiter);

    std::vector<Real> prev_energies;
    bool converged = false;
    int actual_iters = 0;

    const Real effective_breakdown = (config.breakdown_tol > Real(0.0))
        ? config.breakdown_tol
        : mach_eps;

    // 4. Phase 1: Search Loop
    const int max_steps = std::min<int>(config.maxiter, static_cast<int>(dim));
    for (int iter = 0; iter < max_steps; ++iter) {
        actual_iters = iter + 1;

        if constexpr (Traits::stores_basis) {
            VectorView<ExecSpace> v_saved("basis_v", dim);
            Kokkos::deep_copy(v_saved, v_curr);
            basis_vectors.push_back(v_saved);
        }

        H.apply(v_curr, w);
        Real alpha = dot(v_curr, w).real();
        alphas.push_back(alpha);

        axpy(KComplex(-alpha, 0.0), v_curr, w);
        if (iter > 0) {
            axpy(KComplex(-betas.back(), 0.0), v_prev, w);
        }

        // DGKS Full Reorthogonalization (Twice-is-enough)
        if constexpr (Traits::uses_dgks) {
            for (int pass = 0; pass < 2; ++pass) {
                for (const auto& bv : basis_vectors) {
                    KComplex proj = dot(bv, w);
                    axpy(-proj, bv, w);
                }
            }
        }

        Real beta = norm(w);
        if (beta < effective_breakdown) {
            converged = true;
            break;
        }
        if (iter + 1 == static_cast<int>(dim)) {
            converged = true;
            break;
        }
        if (iter + 1 == max_steps) {
            break;
        }
        betas.push_back(beta);

        Kokkos::deep_copy(v_prev, v_curr);
        Kokkos::deep_copy(v_curr, w);
        scal(Real(1.0) / beta, v_curr);

        // Convergence Check
        const int m = static_cast<int>(alphas.size());
        if (m >= target_n_eig && (iter + 1) >= config.min_iterations && (iter + 1) % config.check_interval == 0) {
            if constexpr (Traits::supports_multistate) {
                auto tridiag = linalg::tridiag_eigensystem_full(alphas, betas, m);
                if (!prev_energies.empty()) {
                    bool all_converged = true;
                    for (int k = 0; k < target_n_eig; ++k) {
                        Real diff = std::abs(tridiag.eigenvalues[k] - prev_energies[k]);
                        Real ritz_res = beta * std::abs(tridiag.eigenvectors[k][m - 1]);
                        if (diff > config.tol && ritz_res > config.tol) {
                            all_converged = false;
                            break;
                        }
                    }
                    if (all_converged) {
                        converged = true;
                        break;
                    }
                }
                prev_energies.assign(tridiag.eigenvalues.begin(), tridiag.eigenvalues.begin() + target_n_eig);
            } else {
                auto tridiag = linalg::tridiag_ground_state_full(alphas, betas, m);
                if (!prev_energies.empty()) {
                    if (std::abs(tridiag.energy - prev_energies[0]) < config.tol) {
                        converged = true;
                        break;
                    }
                }
                prev_energies = {tridiag.energy};
            }
        }
    }

    // 5. Final Tridiagonal Eigensystem Solution
    const int final_m = static_cast<int>(alphas.size());
    auto final_tridiag = linalg::tridiag_eigensystem_full(alphas, betas, final_m);

    LanczosResult result;
    result.iterations = actual_iters;
    result.converged = converged;

    const int num_out = std::min<int>(target_n_eig, static_cast<int>(final_tridiag.eigenvalues.size()));
    result.eigenvalues.assign(final_tridiag.eigenvalues.begin(), final_tridiag.eigenvalues.begin() + num_out);
    result.energy = result.eigenvalues.empty() ? Real(0.0) : result.eigenvalues[0];
    result.alphas = alphas;
    result.betas = betas;

    // 6. Phase 2: Vector Reconstruction
    if constexpr (Traits::computes_vector) {
        if (!config.compute_eigenvectors) {
            return result;
        }

        if constexpr (Traits::is_two_pass) {
            // TwoPass: Replay loop strictly for Ground State (k = 0)
            VectorView<ExecSpace> ritz("ritz", dim);
            zero_fill(ritz);

            if (!config.initial_vector.empty()) {
                for (Index i = 0; i < dim; ++i) {
                    v_curr_host(i) = KComplex(config.initial_vector[i].real(), config.initial_vector[i].imag());
                }
                Kokkos::deep_copy(v_curr, v_curr_host);
                normalize(v_curr);
            } else {
                std::mt19937_64 rng(seed);
                std::uniform_real_distribution<Real> dist(-1.0, 1.0);
                for (Index i = 0; i < dim; ++i) v_curr_host(i) = KComplex(dist(rng), dist(rng));
                Kokkos::deep_copy(v_curr, v_curr_host);
                normalize(v_curr);
            }
            zero_fill(v_prev);

            for (int j = 0; j < final_m; ++j) {
                Real y_j = final_tridiag.eigenvectors[0][j];
                axpy(KComplex(y_j, 0.0), v_curr, ritz);

                if (j + 1 == final_m) break; // SpMV elision optimization

                H.apply(v_curr, w);
                axpy(KComplex(-alphas[j], 0.0), v_curr, w);
                if (j > 0) axpy(KComplex(-betas[j - 1], 0.0), v_prev, w);

                Kokkos::deep_copy(v_prev, v_curr);
                Kokkos::deep_copy(v_curr, w);
                scal(Real(1.0) / betas[j], v_curr);
            }

            normalize(ritz);
            copy_device_to_host(ritz, result.eigenvector);
            result.eigenvectors = {result.eigenvector};
        } else if constexpr (Traits::stores_basis) {
            if constexpr (Traits::supports_multistate) {
                // OnePass_DKGS: Arbitrary number of low energy states
                for (int k = 0; k < num_out; ++k) {
                    VectorView<ExecSpace> ritz("ritz", dim);
                    zero_fill(ritz);
                    for (int j = 0; j < final_m; ++j) {
                        Real y_j = final_tridiag.eigenvectors[k][j];
                        axpy(KComplex(y_j, 0.0), basis_vectors[j], ritz);
                    }
                    normalize(ritz);
                    HostVector host_ritz;
                    copy_device_to_host(ritz, host_ritz);
                    result.eigenvectors.push_back(std::move(host_ritz));
                }
                if (!result.eigenvectors.empty()) {
                    result.eigenvector = result.eigenvectors[0];
                }
            } else {
                // OnePass_full: Ground state only (k = 0)
                VectorView<ExecSpace> ritz("ritz", dim);
                zero_fill(ritz);
                for (int j = 0; j < final_m; ++j) {
                    Real y_j = final_tridiag.eigenvectors[0][j];
                    axpy(KComplex(y_j, 0.0), basis_vectors[j], ritz);
                }
                normalize(ritz);
                copy_device_to_host(ritz, result.eigenvector);
                result.eigenvectors = {result.eigenvector};
            }
        }
    }

    return result;
}

} // namespace solvers

// Explicit instantiations for all 4 policies across enabled Kokkos execution spaces
#define INSTANTIATE_LANCZOS(Space) \
    template QKRYLOV_PRECISION_NAMESPACE::LanczosResult solvers::lanczos<solvers::policy::OnePass, Space>(const QKRYLOV_PRECISION_NAMESPACE::MatrixFreeHamiltonian<Space>&, const QKRYLOV_PRECISION_NAMESPACE::LanczosConfig&); \
    template QKRYLOV_PRECISION_NAMESPACE::LanczosResult solvers::lanczos<solvers::policy::OnePass_DKGS, Space>(const QKRYLOV_PRECISION_NAMESPACE::MatrixFreeHamiltonian<Space>&, const QKRYLOV_PRECISION_NAMESPACE::LanczosConfig&); \
    template QKRYLOV_PRECISION_NAMESPACE::LanczosResult solvers::lanczos<solvers::policy::OnePass_full, Space>(const QKRYLOV_PRECISION_NAMESPACE::MatrixFreeHamiltonian<Space>&, const QKRYLOV_PRECISION_NAMESPACE::LanczosConfig&); \
    template QKRYLOV_PRECISION_NAMESPACE::LanczosResult solvers::lanczos<solvers::policy::TwoPass, Space>(const QKRYLOV_PRECISION_NAMESPACE::MatrixFreeHamiltonian<Space>&, const QKRYLOV_PRECISION_NAMESPACE::LanczosConfig&);

#ifdef KOKKOS_ENABLE_SERIAL
INSTANTIATE_LANCZOS(Kokkos::Serial)
#endif
#ifdef KOKKOS_ENABLE_OPENMP
INSTANTIATE_LANCZOS(Kokkos::OpenMP)
#endif
#ifdef KOKKOS_ENABLE_THREADS
INSTANTIATE_LANCZOS(Kokkos::Threads)
#endif
#ifdef KOKKOS_ENABLE_CUDA
INSTANTIATE_LANCZOS(Kokkos::Cuda)
#endif
#ifdef KOKKOS_ENABLE_HIP
INSTANTIATE_LANCZOS(Kokkos::HIP)
#endif
#ifdef KOKKOS_ENABLE_SYCL
INSTANTIATE_LANCZOS(Kokkos::Experimental::SYCL)
#endif

#undef INSTANTIATE_LANCZOS

} // namespace qkrylov
