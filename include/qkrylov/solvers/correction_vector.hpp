#pragma once

#include "qkrylov/core/types.hpp"
#include "qkrylov/linalg/vector_ops.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"

#include <vector>

namespace qkrylov
{

struct CorrectionVectorResult
{
    Vector correction_vector; // |Y(omega + i*eta)>
    double spectral_function; // S(omega)
    int iterations;
    bool converged;
};

// Solve ((H - E0 - omega)^2 + eta^2) |Y> = eta * O |psi0>
// using Matrix-Free Conjugate Gradient (CG) algorithm
CorrectionVectorResult correction_vector_spectral(
    const MatrixFreeHamiltonian& H,
    const Vector& Op_psi0, // O |psi0>
    double E0,
    double omega,
    double eta = 0.1,
    int max_iter = 500,
    double tol = 1e-8
);

}
