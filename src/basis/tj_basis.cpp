#include "qkrylov/core/types.hpp"
#include "qkrylov/basis/tj_basis.hpp"

#include <stdexcept>
#include <bit>
#include <algorithm>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {



TJBasis::TJBasis(
    int N,
    const Sector& sector
)
    : N_(N),
      sector_(sector)
{
    if(N <= 0 || N > 31)
    {
        throw std::runtime_error(
            "TJBasis: N must satisfy 1 <= N <= 31"
        );
    }

    build_basis();
}

Index TJBasis::size() const
{
    return states_.size();
}

StateID TJBasis::state(Index i) const
{
    return states_.at(i);
}

Index TJBasis::index(StateID s) const
{
    auto it = std::lower_bound(states_.begin(), states_.end(), s);

    if(it == states_.end() || *it != s)
        throw std::runtime_error(
            "State not present in basis"
        );

    return static_cast<Index>(std::distance(states_.begin(), it));
}

bool TJBasis::contains(StateID s) const
{
    return std::binary_search(states_.begin(), states_.end(), s);
}

void TJBasis::build_basis()
{
    const StateID dim = StateID(1) << (2 * N_);

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
        // Check for double occupancy
        bool double_occupied = false;
        for(int i = 0; i < N_; ++i)
        {
            if (((s >> (2 * i)) & 1ULL) && ((s >> (2 * i + 1)) & 1ULL))
            {
                double_occupied = true;
                break;
            }
        }

        if (double_occupied) continue;

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
