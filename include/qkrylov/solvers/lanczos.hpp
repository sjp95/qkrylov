#pragma once

#include "qkrylov/core/types.hpp"
#include "qkrylov/linalg/vector_ops.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"
#include "qkrylov/solvers/policy.hpp"

#include <tuple>
#include <vector>
#include <optional>
#include <type_traits>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {

struct LanczosConfig
{
    int n_eig = 1;                               // Number of lowest eigenpairs to compute (OnePass_DKGS)
    int maxiter = 200;                           // Maximum Krylov iterations
    int min_iterations = 1;                      // Minimum iterations before checking convergence
    int check_interval = 1;                      // Convergence check stride
    Real tol = Real(1.0e-12);                    // Energy & Ritz residual convergence tolerance
    Real breakdown_tol = Real(0.0);              // Invariant subspace breakdown tolerance (0 => 4 * eps)
    std::optional<uint64_t> seed = std::nullopt; // Deterministic PRNG seed (default: 123456789ULL)
    HostVector initial_vector = {};              // Warm-starting trial vector
    bool compute_eigenvectors = true;            // Compute state vector(s) when policy supports it
};

struct LanczosResult
{
    Real energy = Real(0.0);                     // Lowest eigenvalue (lambda_0)
    std::vector<Real> eigenvalues = {};          // All computed eigenvalues (size n_eig)
    HostVector eigenvector = {};                 // Lowest eigenvector (psi_0)
    std::vector<HostVector> eigenvectors = {};   // All computed eigenvectors (size n_eig)
    int iterations = 0;                          // Iterations run
    bool converged = false;                      // True if requested state(s) converged
    std::vector<Real> alphas = {};               // Tridiagonal diagonal elements (alpha_0 ... alpha_{m-1})
    std::vector<Real> betas = {};                // Tridiagonal subdiagonal elements (beta_0 ... beta_{m-2})

    // Structured binding support: auto [e, v] = res;
    template <std::size_t I>
    decltype(auto) get() & {
        if constexpr (I == 0) return (energy);
        else if constexpr (I == 1) return (eigenvector);
        else if constexpr (I == 2) return (iterations);
        else if constexpr (I == 3) return (converged);
    }

    template <std::size_t I>
    decltype(auto) get() const & {
        if constexpr (I == 0) return (energy);
        else if constexpr (I == 1) return (eigenvector);
        else if constexpr (I == 2) return (iterations);
        else if constexpr (I == 3) return (converged);
    }

    template <std::size_t I>
    decltype(auto) get() && {
        if constexpr (I == 0) return std::move(energy);
        else if constexpr (I == 1) return std::move(eigenvector);
        else if constexpr (I == 2) return (iterations);
        else if constexpr (I == 3) return (converged);
    }
};

// Aliases for unified result types
using LanczosLowestResult = LanczosResult;
using LanczosLowestConfig = LanczosConfig;

template <std::size_t I>
decltype(auto) get(const LanczosResult& res) {
    return res.template get<I>();
}

template <std::size_t I>
decltype(auto) get(LanczosResult& res) {
    return res.template get<I>();
}

template <std::size_t I>
decltype(auto) get(LanczosResult&& res) {
    return std::move(res).template get<I>();
}

} // namespace QKRYLOV_PRECISION_NAMESPACE

namespace solvers {

using QKRYLOV_PRECISION_NAMESPACE::LanczosConfig;
using QKRYLOV_PRECISION_NAMESPACE::LanczosResult;
using QKRYLOV_PRECISION_NAMESPACE::LanczosLowestConfig;
using QKRYLOV_PRECISION_NAMESPACE::LanczosLowestResult;

template <typename Policy = policy::Default, typename ExecSpace>
LanczosResult lanczos(
    const QKRYLOV_PRECISION_NAMESPACE::MatrixFreeHamiltonian<ExecSpace>& H,
    const LanczosConfig& config = {}
);

template <typename ExecSpace>
inline LanczosResult lanczos_lowest(
    const QKRYLOV_PRECISION_NAMESPACE::MatrixFreeHamiltonian<ExecSpace>& H,
    const LanczosConfig& config = {}
) {
    return lanczos<policy::OnePass_DKGS>(H, config);
}

} // namespace solvers

namespace QKRYLOV_PRECISION_NAMESPACE {

// Idiomatic Convenience Wrappers
template <typename ExecSpace>
inline LanczosResult lanczos_energy(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    const LanczosConfig& config = {}
) {
    return solvers::lanczos<solvers::policy::OnePass>(H, config);
}

template <typename ExecSpace>
inline LanczosResult lanczos_ground_state(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    const LanczosConfig& config = {}
) {
    return solvers::lanczos<solvers::policy::OnePass_DKGS>(H, config);
}

template <typename ExecSpace>
inline LanczosResult lanczos_ground_state(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    int maxiter,
    Real tol = Real(1.0e-12),
    const HostVector& initial_vector = {}
) {
    LanczosConfig cfg;
    cfg.maxiter = maxiter;
    cfg.tol = tol;
    cfg.initial_vector = initial_vector;
    return solvers::lanczos<solvers::policy::OnePass_DKGS>(H, cfg);
}

template <typename ExecSpace>
inline LanczosResult lanczos_two_pass(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    const LanczosConfig& config = {}
) {
    return solvers::lanczos<solvers::policy::TwoPass>(H, config);
}

template <typename ExecSpace>
inline LanczosResult lanczos_two_pass(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    int maxiter,
    Real tol = Real(1.0e-12),
    const HostVector& initial_vector = {}
) {
    LanczosConfig cfg;
    cfg.maxiter = maxiter;
    cfg.tol = tol;
    cfg.initial_vector = initial_vector;
    return solvers::lanczos<solvers::policy::TwoPass>(H, cfg);
}

template <typename ExecSpace>
inline LanczosResult lanczos_lowest(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    int n_eig,
    const LanczosConfig& config = {}
) {
    LanczosConfig cfg = config;
    cfg.n_eig = n_eig;
    return solvers::lanczos<solvers::policy::OnePass_DKGS>(H, cfg);
}

template <typename ExecSpace>
inline LanczosResult lanczos_lowest(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    int n_eig,
    int maxiter,
    Real tol = Real(1.0e-12),
    bool compute_eigenvectors = true,
    const HostVector& initial_vector = {}
) {
    LanczosConfig cfg;
    cfg.n_eig = n_eig;
    cfg.maxiter = maxiter;
    cfg.tol = tol;
    cfg.compute_eigenvectors = compute_eigenvectors;
    cfg.initial_vector = initial_vector;
    return solvers::lanczos<solvers::policy::OnePass_DKGS>(H, cfg);
}

} // namespace QKRYLOV_PRECISION_NAMESPACE

using QKRYLOV_PRECISION_NAMESPACE::lanczos_energy;
using QKRYLOV_PRECISION_NAMESPACE::lanczos_ground_state;
using QKRYLOV_PRECISION_NAMESPACE::lanczos_two_pass;
using QKRYLOV_PRECISION_NAMESPACE::lanczos_lowest;

} // namespace qkrylov

namespace std {

template <>
struct tuple_size<qkrylov::QKRYLOV_PRECISION_NAMESPACE::LanczosResult> : std::integral_constant<std::size_t, 2> {};

template <>
struct tuple_element<0, qkrylov::QKRYLOV_PRECISION_NAMESPACE::LanczosResult> {
    using type = qkrylov::QKRYLOV_PRECISION_NAMESPACE::Real;
};

template <>
struct tuple_element<1, qkrylov::QKRYLOV_PRECISION_NAMESPACE::LanczosResult> {
    using type = qkrylov::QKRYLOV_PRECISION_NAMESPACE::HostVector;
};

} // namespace std
