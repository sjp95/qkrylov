#pragma once

#include "qkrylov/core/types.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"

#include <vector>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {

/// Data stored for a single random sample $r$ in FTLM
template <typename ExecSpace>
struct FTLMSample
{
    Real norm_r = 1.0;
    std::vector<Real> eigenvalues;                       // size M
    std::vector<Real> first_components;                  // Y_{0, m}, size M
    std::vector<std::vector<Real>> ritz_vectors;          // Y_{k, m}, size M x M
    std::vector<VectorView<ExecSpace>> krylov_basis;     // |v_k>, M vectors of dimension D
};

/// Result of an FTLM calculation holding all random sample data
template <typename ExecSpace>
struct FTLMResult
{
    Real beta = 1.0;
    Real partition_function_val = 0.0;
    Real internal_energy_val = 0.0;
    Real specific_heat_val = 0.0;
    Index hilbert_dim = 0;

    std::vector<FTLMSample<ExecSpace>> samples;

    // Partition function Z(beta)
    Real partition_function(Real b) const;
    std::vector<Real> partition_function(const std::vector<Real>& betas) const;

    // Internal energy <H>_beta
    Real internal_energy(Real b) const;
    std::vector<Real> internal_energy(const std::vector<Real>& betas) const;

    // Expectation value <A>_beta for an arbitrary operator A
    Real expectation_value(const MatrixFreeHamiltonian<ExecSpace>& A, Real b) const;
    std::vector<Real> expectation_value(const MatrixFreeHamiltonian<ExecSpace>& A, const std::vector<Real>& betas) const;
};

/// Run FTLM sampling on Hamiltonian H
template <typename ExecSpace>
FTLMResult<ExecSpace> ftlm(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    Real beta = 1.0,
    int n_random = 50,
    int n_steps = 100
);

/// Finite-Temperature Dynamical Correlation Function S_AB(omega, beta)
template <typename ExecSpace>
std::vector<Real> ftlm_dynamical_correlation(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    const MatrixFreeHamiltonian<ExecSpace>& A,
    const MatrixFreeHamiltonian<ExecSpace>& B,
    Real beta,
    int n_random,
    int n_steps_thermal,
    int n_steps_dyn,
    const std::vector<Real>& omegas,
    Real eta = 0.1
);

}

}
