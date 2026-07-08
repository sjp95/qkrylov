#pragma once

#include "qkrylov/basis/basis.hpp"
#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/sites/site.hpp"

#include <memory>
#include <vector>

namespace qkrylov
{

class CUDAHamiltonian
{
public:

    CUDAHamiltonian(
        std::shared_ptr<Basis> basis,
        std::shared_ptr<Site> site,
        const OpSum& ops
    );

    ~CUDAHamiltonian();

    void apply(
        const std::vector<Complex>& x,
        std::vector<Complex>& y
    ) const;

    Index dimension() const;

private:

    // Device-side data would go here (e.g. basis states, operator encoding)
    void* d_basis = nullptr;
    void* d_ops = nullptr;

    std::shared_ptr<Basis> basis_;
};

}
