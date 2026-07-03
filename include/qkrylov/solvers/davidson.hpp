#pragma once

#include "qkrylov/core/types.hpp"
#include "qkrylov/linalg/vector_ops.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"

#include <vector>

namespace qkrylov
{

struct DavidsonResult
{
    std::vector<double> eigenvalues;
    std::vector<Vector> eigenvectors;
};

DavidsonResult davidson_lowest(
    const MatrixFreeHamiltonian& H,
    int n_eig = 1,
    int max_subspace = 20,
    double tol = 1.0e-8
);

}
