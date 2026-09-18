#pragma once

#include "qkrylov/core/types.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"
#include "qkrylov/solvers/policy.hpp"

#include <vector>
#include <string>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {

enum class FTLMWorkflow {
    Streamed, // Two-pass on-the-fly accumulation, bounded O(M^2 * N_obs) host RAM
    Cached    // Two-stage decoupled sampling, O(R * M^2 * N_obs) host RAM, post-hoc tunable
};

struct ProjectedOperator
{
    std::string name = "";
    std::vector<Complex> matrix = {}; // Flattened row-major M x M matrix: matrix[j * M + k] = <v_j | O | v_k>
    bool is_hermitian = true;
};

struct FTLMKrylovSample
{
    Index dimension = 0;                              // Hilbert space dimension D
    Real norm = Real(1.0);                            // Unit sphere normalization
    std::vector<Real> eigenvalues = {};               // Ritz eigenvalues of T_M (ascending order), size M
    std::vector<Real> first_components = {};          // y_{0m} components, size M
    std::vector<std::vector<Real>> eigenvectors = {}; // Y matrix (M x M): eigenvectors[m][j]
    std::vector<ProjectedOperator> projected_operators = {}; // Projected matrices for K observables
};

struct FTLMSweepResult
{
    Index dimension = 0;                                          // Hilbert space dimension D
    std::vector<Real> beta_grid = {};
    std::vector<Real> partition_functions = {};                   // Z(beta) = D * Z_bar * exp(-beta * E_min)
    std::vector<Real> free_energies = {};                         // F(beta) = -ln(Z)/beta
    std::vector<Real> internal_energies = {};                     // <H>(beta)
    std::vector<Real> specific_heats = {};                        // Cv(beta) = beta^2 (<H^2> - <H>^2)
    std::vector<Real> entropies = {};                             // S(beta) (thermodynamic integration)
    std::vector<Real> effective_samples = {};                     // R_eff(beta) diagnostic
    std::vector<std::vector<Complex>> observable_expectations = {}; // [obs_idx][beta_idx] (Complex)
    std::vector<std::vector<Real>> observable_errors = {};        // [obs_idx][beta_idx] (linearized ratio std err)
};

struct FTLMResult
{
    Index dimension = 0;
    Real beta = Real(0.0);
    Real partition_function = Real(0.0);
    Real free_energy = Real(0.0);
    Real internal_energy = Real(0.0);
    Real specific_heat = Real(0.0);
    Real entropy = Real(0.0);
    Real effective_samples = Real(0.0);
    std::vector<Complex> observable_expectations = {};
    std::vector<Real> observable_errors = {};
};

// Mode 2 / Stage 1: Krylov Subspace Sampling (Cached mode)
template <typename ExecSpace, typename Policy = solvers::policy::OnePass_full>
std::vector<FTLMKrylovSample> ftlm_sample(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    const std::vector<MatrixFreeHamiltonian<ExecSpace>>& observables = {},
    int n_random = 50,
    int n_steps = 100,
    uint64_t seed = 42
);

// Mode 2 / Stage 2: Fast Boltzmann Multi-Temperature Evaluator (Cached mode)
FTLMSweepResult ftlm_evaluate_sweep(
    const std::vector<FTLMKrylovSample>& samples,
    const std::vector<Real>& beta_grid,
    Index dimension = 0
);

// Mode 1: Streamed Two-Pass Multi-Temperature Sweep (Memory-bounded production mode)
template <typename ExecSpace, typename Policy = solvers::policy::OnePass_full>
FTLMSweepResult ftlm_sweep_streamed(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    const std::vector<Real>& beta_grid,
    const std::vector<MatrixFreeHamiltonian<ExecSpace>>& observables = {},
    int n_random = 50,
    int n_steps = 100,
    uint64_t seed = 42
);

// End-to-End Multi-Temperature Sweep (Defaults to Streamed for maximum efficiency)
template <typename ExecSpace, typename Policy = solvers::policy::OnePass_full>
FTLMSweepResult ftlm_sweep(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    const std::vector<Real>& beta_grid,
    const std::vector<MatrixFreeHamiltonian<ExecSpace>>& observables = {},
    int n_random = 50,
    int n_steps = 100,
    uint64_t seed = 42,
    FTLMWorkflow workflow = FTLMWorkflow::Streamed
);

// Single-temperature evaluation wrapper with observables
template <typename ExecSpace, typename Policy = solvers::policy::OnePass_full>
FTLMResult ftlm(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    Real beta,
    const std::vector<MatrixFreeHamiltonian<ExecSpace>>& observables,
    int n_random = 50,
    int n_steps = 100,
    uint64_t seed = 42
);

// Single-temperature evaluation wrapper without observables
template <typename ExecSpace, typename Policy = solvers::policy::OnePass_full>
inline FTLMResult ftlm(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    Real beta,
    int n_random = 50,
    int n_steps = 100,
    uint64_t seed = 42
) {
    return ftlm<ExecSpace, Policy>(H, beta, {}, n_random, n_steps, seed);
}

} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov
