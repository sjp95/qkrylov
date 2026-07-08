#include "qkrylov/sites/tj_site.hpp"

#include <stdexcept>
#include <bit>

namespace qkrylov
{

bool TJSite::occupied_up(
    StateID state,
    int site
)
{
    return (state >> (2 * site)) & 1ULL;
}

bool TJSite::occupied_dn(
    StateID state,
    int site
)
{
    return (state >> (2 * site + 1)) & 1ULL;
}

double TJSite::phase_up(
    StateID state,
    int site
)
{
    StateID mask = (1ULL << (2 * site)) - 1;
    int count = std::popcount(state & mask);
    return (count % 2 == 0) ? 1.0 : -1.0;
}

double TJSite::phase_dn(
    StateID state,
    int site
)
{
    StateID mask = (1ULL << (2 * site + 1)) - 1;
    int count = std::popcount(state & mask);
    return (count % 2 == 0) ? 1.0 : -1.0;
}

LocalAction TJSite::apply(
    const std::string& op,
    StateID local_state
) const
{
    LocalAction a;

    const bool up = local_state & 1ULL;
    const bool dn = (local_state >> 1) & 1ULL;

    if(op == "Nup")
    {
        a.valid = true;
        a.new_state = local_state;
        a.matrix_element = up ? 1.0 : 0.0;
        return a;
    }

    if(op == "Ndn")
    {
        a.valid = true;
        a.new_state = local_state;
        a.matrix_element = dn ? 1.0 : 0.0;
        return a;
    }

    if(op == "CUp")
    {
        if(!up) return a;
        a.valid = true;
        a.new_state = local_state & ~1ULL;
        a.matrix_element = 1.0;
        return a;
    }

    if(op == "CdagUp")
    {
        if(up || dn) return a; // Forbidden if already occupied
        a.valid = true;
        a.new_state = 1ULL;
        a.matrix_element = 1.0;
        return a;
    }

    if(op == "CDn")
    {
        if(!dn) return a;
        a.valid = true;
        a.new_state = local_state & ~2ULL;
        a.matrix_element = 1.0;
        return a;
    }

    if(op == "CdagDn")
    {
        if(dn || up) return a; // Forbidden if already occupied
        a.valid = true;
        a.new_state = 2ULL;
        a.matrix_element = 1.0;
        return a;
    }

    throw std::runtime_error(
        "Unknown TJ operator: " + op
    );
}

}
