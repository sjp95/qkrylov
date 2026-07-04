#pragma once

#include "qkrylov/basis/basis.hpp"
#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/sites/site.hpp"

#include <memory>
#include <vector>

namespace qkrylov
{

class MatrixFreeHamiltonian
{
public:

    using Vector =
        std::vector<Complex>;

    MatrixFreeHamiltonian(
        std::shared_ptr<Basis> basis,
        std::shared_ptr<Site> site,
        const OpSum& ops
    );

    void apply(
        const Vector& x,
        Vector& y
    ) const;

    Vector diagonal() const;

    Index dimension() const;

private:

    std::shared_ptr<Basis> basis_;

    std::shared_ptr<Site> site_;

    OpSum ops_;
};

}
