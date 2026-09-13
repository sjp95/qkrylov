#include "qkrylov/core/types.hpp"
#include "qkrylov/basis/hubbard_basis.hpp"

#include <stdexcept>
#include <bit>
#include <algorithm>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {



HubbardBasis::HubbardBasis(
    int N,
    const Sector& sector
)
    : N_(N),
      sector_(sector)
{
    if(N <= 0 || N > 31) // N sites means 2N bits
    {
        throw std::runtime_error(
            "HubbardBasis: N must satisfy 1 <= N <= 31"
        );
    }

    if(sector_.use_nup || sector_.use_ndn)
        build_nup_ndn_basis();
    else
        build_full_basis();
}

Index HubbardBasis::size() const
{
    return states_.size();
}

StateID HubbardBasis::state(Index i) const
{
    return states_.at(i);
}

Index HubbardBasis::index(StateID s) const
{
    if (!sector_.use_nup && !sector_.use_ndn) {
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

bool HubbardBasis::contains(StateID s) const
{
    if (!sector_.use_nup && !sector_.use_ndn) {
        return s < static_cast<StateID>(states_.size());
    }
    return std::binary_search(states_.begin(), states_.end(), s);
}

void HubbardBasis::build_full_basis()
{
    const StateID dim = StateID(1) << (2 * N_);

    states_.reserve(dim);

    for(StateID s = 0; s < dim; ++s)
    {
        states_.push_back(s);
    }
}

void HubbardBasis::build_nup_ndn_basis()
{
    const StateID dim =
        StateID(1) << (2 * N_);

    states_.reserve(dim / 2);

    StateID up_mask = 0;
    StateID dn_mask = 0;
    for(int i = 0; i < N_; ++i)
    {
        up_mask |= (1ULL << (2 * i));
        dn_mask |= (1ULL << (2 * i + 1));
    }

    for(StateID s = 0; s < dim; ++s)
    {
        bool match_up = !sector_.use_nup || (popcount(s & up_mask) == sector_.nup);
        bool match_dn = !sector_.use_ndn || (popcount(s & dn_mask) == sector_.ndn);

        if(match_up && match_dn)
        {
            states_.push_back(s);
        }
    }
    states_.shrink_to_fit();
}



} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov
