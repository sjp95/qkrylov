#include "qkrylov/core/types.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"

#include <stdexcept>
#include <bit>
#include <algorithm>

namespace qkrylov {



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
    if (!sector_.use_sz) {
        return Index(1) << N_;
    }
    return states_.size();
}

StateID SpinHalfBasis::state(Index i) const
{
    if (!sector_.use_sz) {
        const Index dim = Index(1) << N_;
        if (i >= dim) {
            throw std::out_of_range("SpinHalfBasis::state: index out of range");
        }
        return static_cast<StateID>(i);
    }
    return states_.at(i);
}

Index SpinHalfBasis::index(StateID s) const
{
    if (!sector_.use_sz) {
        const StateID dim = StateID(1) << N_;
        if (s < dim) {
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
        const StateID dim = StateID(1) << N_;
        return s < dim;
    }
    return std::binary_search(states_.begin(), states_.end(), s);
}

void SpinHalfBasis::build_full_basis()
{
    // Implicit indexing: states_ remains empty, zero RAM allocated.
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



} // namespace qkrylov