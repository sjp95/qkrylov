#include "qkrylov/basis/fermion_basis.hpp"

#include <stdexcept>
#include <bit>

namespace qkrylov
{

FermionBasis::FermionBasis(
    int N,
    const Sector& sector
)
    : N_(N),
      sector_(sector)
{
    if(N <= 0 || N > 63)
    {
        throw std::runtime_error(
            "FermionBasis: N must satisfy 1 <= N <= 63"
        );
    }

    if(sector_.use_n)
        build_n_basis();
    else
        build_full_basis();
}

Index FermionBasis::size() const
{
    return states_.size();
}

StateID FermionBasis::state(Index i) const
{
    return states_.at(i);
}

Index FermionBasis::index(StateID s) const
{
    auto it = lookup_.find(s);

    if(it == lookup_.end())
        throw std::runtime_error(
            "State not present in basis"
        );

    return it->second;
}

bool FermionBasis::contains(StateID s) const
{
    return lookup_.contains(s);
}

void FermionBasis::build_full_basis()
{
    const StateID dim = StateID(1) << N_;

    states_.reserve(dim);

    for(StateID s = 0; s < dim; ++s)
    {
        lookup_[s] = states_.size();
        states_.push_back(s);
    }
}

void FermionBasis::build_n_basis()
{
    const StateID dim =
        StateID(1) << N_;

    states_.reserve(dim);

    for(StateID s = 0; s < dim; ++s)
    {
        if(std::popcount(s) == sector_.n)
        {
            lookup_[s] =
                states_.size();

            states_.push_back(s);
        }
    }
}

} // namespace qkrylov
