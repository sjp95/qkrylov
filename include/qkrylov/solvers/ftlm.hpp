#pragma once

#include "qkrylov/core/types.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"

#include <vector>

namespace qkrylov
{

struct FTLMResult
{
    double beta;
    double partition_function = 0.0;
    double internal_energy = 0.0;
    double specific_heat = 0.0;
    // We can add more observables later
};

FTLMResult ftlm(
    const MatrixFreeHamiltonian& H,
    double beta,
    int n_random = 50,
    int n_steps = 100
);

}
