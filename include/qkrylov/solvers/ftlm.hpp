#pragma once

#include "qkrylov/core/types.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"
#include "qkrylov/solvers/policy.hpp"

#include <vector>
#include <string>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {

struct ProjectedOperator
{
    std::string name = "";
    std::vector<Complex> matrix = {}; // Flattened row-major M x M matrix: matrix[j * M + k] = <v_j | O | v_k>
    bool is_hermitian = true;
};

struct FTLMKrylovSample
{
    Real norm = Real(0.0);
    std::vector<Real> eigenvalues = {};               // Ritz eigenvalues of T_M (ascending order), size M
    std::vector<Real> first_components = {};          // y_{0m} components, size M
    std::vector<std::vector<Real>> eigenvectors = {}; // Y matrix (M x M): eigenvectors[m][j]
    std::vector<ProjectedOperator> projected_operators = {}; // Projected matrices for K observables
};

struct FTLMSweepResult
{
    std::vector<Real> beta_grid = {};
    std::vector<Real> partition_functions = {};                   // Z(beta)
    std::vector<Real> free_energies = {};                         // F(beta) = -ln(Z)/beta
    std::vector<Real> internal_energies = {};                     // <H>(beta)
    std::vector<Real> specific_heats = {};                        // Cv(beta) = beta^2 (<H^2> - <H>^2)
    std::vector<Real> entropies = {};                             // S(beta) = beta (E - F)
    std::vector<std::vector<Real>> observable_expectations = {};  // [obs_idx][beta_idx]
    std::vector<std::vector<Real>> observable_errors = {};        // [obs_idx][beta_idx] (sample standard error)
};

struct FTLMResult
{
    Real beta = Real(0.0);
    Real partition_function = Real(0.0);
    Real free_energy = Real(0.0);
    Real internal_energy = Real(0.0);
    Real specific_heat = Real(0.0);
    Real entropy = Real(0.0);
    std::vector<Real> observable_expectations = {};
    std::vector<Real> observable_errors = {};
};

// Stage 1: Krylov Subspace Sampling
template <typename ExecSpace, typename Policy = solvers::policy::OnePass_full>
std::vector<FTLMKrylovSample> ftlm_sample(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    const std::vector<MatrixFreeHamiltonian<ExecSpace>>& observables = {},
    int n_random = 50,
    int n_steps = 100,
    uint64_t seed = 42
);

// Stage 2: Fast Boltzmann Multi-Temperature Evaluator
FTLMSweepResult ftlm_evaluate_sweep(
    const std::vector<FTLMKrylovSample>& samples,
    const std::vector<Real>& beta_grid
);

// End-to-End Multi-Temperature Sweep
template <typename ExecSpace, typename Policy = solvers::policy::OnePass_full>
FTLMSweepResult ftlm_sweep(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    const std::vector<Real>& beta_grid,
    const std::vector<MatrixFreeHamiltonian<ExecSpace>>& observables = {},
    int n_random = 50,
    int n_steps = 100,
    uint64_t seed = 42
);

// Backward-compatible single-point evaluation wrapper
template <typename ExecSpace, typename Policy = solvers::policy::OnePass_full>
FTLMResult ftlm(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    Real beta,
    int n_random = 50,
    int n_steps = 100,
    uint64_t seed = 42
);

} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov
