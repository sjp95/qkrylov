#include "qkrylov/basis/hubbard_basis.hpp"

#include <stdexcept>
#include <bit>

namespace qkrylov
{

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
    auto it = lookup_.find(s);

    if(it == lookup_.end())
        throw std::runtime_error(
            "State not present in basis"
        );

    return it->second;
}

bool HubbardBasis::contains(StateID s) const
{
    return lookup_.contains(s);
}

void HubbardBasis::build_full_basis()
{
    const StateID dim = StateID(1) << (2 * N_);

    states_.reserve(dim);

    for(StateID s = 0; s < dim; ++s)
    {
        lookup_[s] = states_.size();
        states_.push_back(s);
    }
}

void HubbardBasis::build_nup_ndn_basis()
{
    const StateID dim =
        StateID(1) << (2 * N_);

    states_.reserve(dim);

    for(StateID s = 0; s < dim; ++s)
    {
        StateID up_mask = 0;
        StateID dn_mask = 0;
        for(int i=0; i<N_; ++i)
        {
            up_mask |= (1ULL << (2*i));
            dn_mask |= (1ULL << (2*i + 1));
        }

        bool match_up = !sector_.use_nup || (std::popcount(s & up_mask) == sector_.nup);
        bool match_dn = !sector_.use_ndn || (std::popcount(s & dn_mask) == sector_.ndn);

        if(match_up && match_dn)
        {
            lookup_[s] =
                states_.size();

            states_.push_back(s);
        }
    }
}

} // namespace qkrylov
