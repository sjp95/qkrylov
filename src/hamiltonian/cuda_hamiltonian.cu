#include "qkrylov/hamiltonian/cuda_hamiltonian.hpp"
#include <iostream>

namespace qkrylov
{

CUDAHamiltonian::CUDAHamiltonian(
    std::shared_ptr<Basis> basis,
    std::shared_ptr<Site> site,
    const OpSum& ops
)
    : basis_(std::move(basis))
{
    std::cout << "CUDA acceleration is currently a scaffolding. Device allocation would happen here.\n";
}

CUDAHamiltonian::~CUDAHamiltonian()
{
    // Free device memory
}

void CUDAHamiltonian::apply(
    const std::vector<Complex>& x,
    std::vector<Complex>& y
) const
{
    std::cout << "CUDA Hamiltonian apply would execute kernels on the GPU.\n";
    // Placeholder: copy to GPU, run kernel, copy back
}

Index CUDAHamiltonian::dimension() const
{
    return basis_->size();
}

}
