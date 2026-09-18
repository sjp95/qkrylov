#pragma once

#include "qkrylov/core/types.hpp"
#include "qkrylov/linalg/vector_ops.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"

#include <vector>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {

struct DynamicsResult
{
    std::vector<Real> alphas;
    std::vector<Real> betas;
    Real norm_phi0;
};

struct RealTimeResult
{
    std::vector<Real> time_grid = {};
    std::vector<Complex> survival_probabilities = {};             // <psi(0)|psi(t)>
    std::vector<std::vector<Complex>> observable_expectations = {}; // [obs_idx][time_idx]
};

struct FTLMDynamicsResult
{
    Real beta = Real(0.0);
    std::vector<Real> time_grid = {};
    std::vector<Complex> correlations = {};    // C_AB(t) = <A(t) B(0)>_beta
    std::vector<Real> correlation_errors = {}; // Linearized ratio standard error
};

// Compute Continued Fraction coefficients starting from vector phi0 (frequency-domain spectral function)
template <typename ExecSpace>
DynamicsResult continued_fraction_coeffs(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    const HostVector& phi0,
    int n_iter = 100
);

// Helper function to evaluate S(omega) from coefficients
// S(omega) = -1/pi * Im <phi0 | (omega - H + E0 + i*eta)^-1 | phi0>
Real evaluate_spectral_function(
    const Real* alphas,
    const Real* betas,
    size_t n,
    Real norm_phi0,
    Real omega,
    Real E0,
    Real eta = 0.1
);

// Pure-state real-time evolution: |psi(t)> = exp(-i*H*t) |psi(0)>
template <typename ExecSpace>
RealTimeResult time_evolve(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    const HostVector& psi0,
    const std::vector<Real>& time_grid,
    const std::vector<MatrixFreeHamiltonian<ExecSpace>>& observables = {},
    int n_steps = 50
);

// Finite-temperature real-time dynamical correlator: C_AB(t) = <A(t) B(0)>_beta
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

} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov
