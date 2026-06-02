#include "qkrylov/basis/spinhalf_basis.hpp"

#include <stdexcept>
#include <bit>

namespace qkrylov
{

SpinHalfBasis::SpinHalfBasis(
    int N,
    const Sector& sector
)
    : N_(N),
      sector_(sector)
{
    if(N <= 0 || N > 63)
    {
        throw std::runtime_error(
            "SpinHalfBasis: N must satisfy 1 <= N <= 63"
        );
    }

    if(sector_.use_sz)
        build_sz_basis();
    else
        build_full_basis();
}

Index SpinHalfBasis::size() const
{
    return states_.size();
}

StateID SpinHalfBasis::state(Index i) const
{
    return states_.at(i);
}

Index SpinHalfBasis::index(StateID s) const
{
    auto it = lookup_.find(s);

    if(it == lookup_.end())
        throw std::runtime_error(
            "State not present in basis"
        );

    return it->second;
}

bool SpinHalfBasis::contains(StateID s) const
{
    return lookup_.contains(s);
}

void SpinHalfBasis::build_full_basis()
{
    const StateID dim = StateID(1) << N_;

    states_.reserve(dim);

    for(StateID s = 0; s < dim; ++s)
    {
        lookup_[s] = states_.size();
        states_.push_back(s);
    }
}

int SpinHalfBasis::compute_sz2(
    StateID state,
    int N
)
{
    const int nup =
        std::popcount(state);

    const int ndown =
        N - nup;

    return nup - ndown;
}

void SpinHalfBasis::build_sz_basis()
{
    const StateID dim =
        StateID(1) << N_;

    states_.reserve(dim);

    for(StateID s = 0; s < dim; ++s)
    {
        if(compute_sz2(s, N_) ==
           sector_.sz2)
        {
            lookup_[s] =
                states_.size();

            states_.push_back(s);
        }
    }
}

} // namespace qkrylov