#pragma once

#include "qkrylov/core/types.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"
#include <Kokkos_Core.hpp>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {

struct CorrectionVectorResult {
    HostVector correction_vector;
    Real spectral_function = 0.0;
    int iterations = 0;
    bool converged = false;
};

template <typename ExecSpace = Kokkos::DefaultExecutionSpace>
CorrectionVectorResult correction_vector_spectral(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    const HostVector& Op_psi0,
    Real E0,
    Real omega,
    Real eta = 0.1,
    int max_iter = 500,
    Real tol = 1e-8
);

} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov
