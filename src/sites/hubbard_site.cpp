#include "qkrylov/sites/hubbard_site.hpp"

#include <stdexcept>
#include <bit>

namespace qkrylov
{

// We'll use a mapping where site i has:
// Up electron at bit 2*i
// Down electron at bit 2*i + 1

bool HubbardSite::occupied_up(
    StateID state,
    int site
)
{
    return (state >> (2 * site)) & 1ULL;
}

bool HubbardSite::occupied_dn(
    StateID state,
    int site
)
{
    return (state >> (2 * site + 1)) & 1ULL;
}

double HubbardSite::phase_up(
    StateID state,
    int site
)
{
    StateID mask = (1ULL << (2 * site)) - 1;
    int count = std::popcount(state & mask);
    return (count % 2 == 0) ? 1.0 : -1.0;
}

double HubbardSite::phase_dn(
    StateID state,
    int site
)
{
    StateID mask = (1ULL << (2 * site + 1)) - 1;
    int count = std::popcount(state & mask);
    return (count % 2 == 0) ? 1.0 : -1.0;
}

LocalAction HubbardSite::apply(
    const std::string& op,
    int site,
    StateID state
) const
{
    LocalAction a;

    const bool up = occupied_up(state, site);
    const bool dn = occupied_dn(state, site);

    if(op == "Nup")
    {
        a.valid = true;
        a.new_state = state;
        a.matrix_element = up ? 1.0 : 0.0;
        return a;
    }

    if(op == "Ndn")
    {
        a.valid = true;
        a.new_state = state;
        a.matrix_element = dn ? 1.0 : 0.0;
        return a;
    }

    if(op == "Nupdn")
    {
        a.valid = true;
        a.new_state = state;
        a.matrix_element = (up && dn) ? 1.0 : 0.0;
        return a;
    }

    if(op == "CUp")
    {
        if(!up) return a;
        a.valid = true;
        a.new_state = state & ~(1ULL << (2 * site));
        a.matrix_element = phase_up(state, site);
        return a;
    }

    if(op == "CdagUp")
    {
        if(up) return a;
        a.valid = true;
        a.new_state = state | (1ULL << (2 * site));
        a.matrix_element = phase_up(state, site);
        return a;
    }

    if(op == "CDn")
    {
        if(!dn) return a;
        a.valid = true;
        a.new_state = state & ~(1ULL << (2 * site + 1));
        a.matrix_element = phase_dn(state, site);
        return a;
    }

    if(op == "CdagDn")
    {
        if(dn) return a;
        a.valid = true;
        a.new_state = state | (1ULL << (2 * site + 1));
        a.matrix_element = phase_dn(state, site);
        return a;
    }

    throw std::runtime_error(
        "Unknown Hubbard operator: " + op
    );
}

}
