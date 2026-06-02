#pragma once

#include "qkrylov/core/types.hpp"
#include "qkrylov/linalg/vector_ops.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"

namespace qkrylov
{

struct LanczosResult
{
    double energy = 0.0;

    Vector eigenvector;
};

LanczosResult lanczos_ground_state(
    const MatrixFreeHamiltonian& H,
    int maxiter = 200,
    double tol = 1.0e-12
);

}
