#pragma once

#include "qkrylov/core/types.hpp"
#include "qkrylov/linalg/vector_ops.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {


struct LanczosResult
{
    Real energy = 0.0;
    int iterations = 0;
    bool converged = false;

    HostVector eigenvector;
};

template <typename ExecSpace>
LanczosResult lanczos_ground_state(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    int maxiter = 200,
    Real tol = 1.0e-12,
    bool two_pass = true
);

}

}
