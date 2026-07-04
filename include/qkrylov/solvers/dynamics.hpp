#pragma once

#include "qkrylov/core/types.hpp"
#include "qkrylov/linalg/vector_ops.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"

#include <vector>

namespace qkrylov
{

struct DynamicsResult
{
    std::vector<double> alphas;
    std::vector<double> betas;
    double norm_phi0;
};

// Compute the Continued Fraction coefficients starting from a vector phi0
DynamicsResult continued_fraction_coeffs(
    const MatrixFreeHamiltonian& H,
    const Vector& phi0,
    int n_iter = 100
);

// Helper function to evaluate S(omega) from coefficients
// S(omega) = -1/pi * Im <phi0 | (omega - H + E0 + i*eta)^-1 | phi0>
double evaluate_spectral_function(
    const DynamicsResult& res,
    double omega,
    double E0,
    double eta = 0.1
);

}
