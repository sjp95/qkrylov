#pragma once

#include "qkrylov/core/types.hpp"
#include "qkrylov/linalg/vector_ops.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"
#include "qkrylov/solvers/policy.hpp"

#include <tuple>
#include <type_traits>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {

struct LanczosConfig
{
    int maxiter = 200;
    Real tol = 1.0e-12;
    HostVector initial_vector = {};
};

struct LanczosResult
{
    Real energy = 0.0;
    int iterations = 0;
    bool converged = false;

    HostVector eigenvector;

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

struct LanczosLowestResult
{
    std::vector<Real> eigenvalues;
    std::vector<HostVector> eigenvectors;
    int iterations = 0;
    bool converged = false;
};

struct LanczosLowestConfig
{
    int n_eig = 1;
    int maxiter = 200;
    Real tol = 1.0e-12;
    bool compute_eigenvectors = true;
    HostVector initial_vector = {};
};

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
LanczosLowestResult lanczos_lowest(
    const QKRYLOV_PRECISION_NAMESPACE::MatrixFreeHamiltonian<ExecSpace>& H,
    const LanczosLowestConfig& config = {}
);

} // namespace solvers

namespace QKRYLOV_PRECISION_NAMESPACE {

// Convenience functions
template <typename ExecSpace>
inline LanczosResult lanczos_ground_state(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    int maxiter = 200,
    Real tol = 1.0e-12,
    const HostVector& initial_vector = {}
) {
    return solvers::lanczos<solvers::policy::Default>(H, {maxiter, tol, initial_vector});
}

template <typename ExecSpace>
inline LanczosResult lanczos_two_pass(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    int maxiter = 200,
    Real tol = 1.0e-12,
    const HostVector& initial_vector = {}
) {
    return solvers::lanczos<solvers::policy::TwoPass>(H, {maxiter, tol, initial_vector});
}

template <typename ExecSpace>
inline LanczosLowestResult lanczos_lowest(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    int n_eig = 1,
    int maxiter = 200,
    Real tol = 1.0e-12,
    bool compute_eigenvectors = true,
    const HostVector& initial_vector = {}
) {
    return solvers::lanczos_lowest(H, {n_eig, maxiter, tol, compute_eigenvectors, initial_vector});
}

} // namespace QKRYLOV_PRECISION_NAMESPACE

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
