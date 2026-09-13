#include "qkrylov/core/types.hpp"
#include "qkrylov/basis/fermion_basis.hpp"

#include <stdexcept>
#include <bit>
#include <algorithm>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {



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
    if (!sector_.use_n) {
        if (s < static_cast<StateID>(states_.size())) {
            return static_cast<Index>(s);
        }
        throw std::runtime_error("State not present in basis");
    }

    auto it = std::lower_bound(states_.begin(), states_.end(), s);
    if (it == states_.end() || *it != s) {
        throw std::runtime_error("State not present in basis");
    }
    return static_cast<Index>(std::distance(states_.begin(), it));
}

bool FermionBasis::contains(StateID s) const
{
    if (!sector_.use_n) {
        return s < static_cast<StateID>(states_.size());
    }
    return std::binary_search(states_.begin(), states_.end(), s);
}

void FermionBasis::build_full_basis()
{
    const StateID dim = StateID(1) << N_;

    states_.reserve(dim);

    for(StateID s = 0; s < dim; ++s)
    {
        states_.push_back(s);
    }
}

void FermionBasis::build_n_basis()
{
    const StateID dim =
        StateID(1) << N_;

    states_.reserve(dim / 2);

    for(StateID s = 0; s < dim; ++s)
    {
        if(popcount(s) == sector_.n)
        {
            states_.push_back(s);
        }
    }
    states_.shrink_to_fit();
}



} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov
