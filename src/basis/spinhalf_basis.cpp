#include "qkrylov/core/types.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"

#include <stdexcept>
#include <bit>
#include <algorithm>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {



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
    if (!sector_.use_sz) {
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

bool SpinHalfBasis::contains(StateID s) const
{
    if (!sector_.use_sz) {
        return s < static_cast<StateID>(states_.size());
    }
    return std::binary_search(states_.begin(), states_.end(), s);
}

void SpinHalfBasis::build_full_basis()
{
    const StateID dim = StateID(1) << N_;

    states_.reserve(dim);

    for(StateID s = 0; s < dim; ++s)
    {
        states_.push_back(s);
    }
}

int SpinHalfBasis::compute_sz2(
    StateID state,
    int N
)
{
    const int nup =
        popcount(state);

    const int ndown =
        N - nup;

    return nup - ndown;
}

void SpinHalfBasis::build_sz_basis()
{
    const StateID dim =
        StateID(1) << N_;

    states_.reserve(dim / 2);

    for(StateID s = 0; s < dim; ++s)
    {
        if(compute_sz2(s, N_) ==
           sector_.sz2)
        {
            states_.push_back(s);
        }
    }
    states_.shrink_to_fit();
}



} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov