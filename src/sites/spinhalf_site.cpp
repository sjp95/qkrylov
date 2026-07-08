#include "qkrylov/sites/spinhalf_site.hpp"

#include <stdexcept>

namespace qkrylov
{

bool SpinHalfSite::spin_up(
    StateID state,
    int site
)
{
    return (state >> site) & 1ULL;
}

LocalAction SpinHalfSite::apply(
    const std::string& op,
    StateID local_state
) const
{
    LocalAction a;

    const bool up = local_state & 1ULL;

    //
    // Sz
    //
    if(op == "Sz")
    {
        a.valid = true;
        a.new_state = local_state;

        a.matrix_element =
            up ? 0.5 : -0.5;

        return a;
    }

    //
    // Sp
    //
    if(op == "Sp")
    {
        if(up)
            return a;

        a.valid = true;

        a.new_state = 1ULL;

        a.matrix_element = 1.0;

        return a;
    }

    //
    // Sm
    //
    if(op == "Sm")
    {
        if(!up)
            return a;

        a.valid = true;

        a.new_state = 0ULL;

        a.matrix_element = 1.0;

        return a;
    }

    //
    // Sx
    //
    if(op == "Sx")
    {
        a.valid = true;

        a.new_state = local_state ^ 1ULL;

        a.matrix_element = 0.5;

        return a;
    }

    //
    // Sy
    //
    if(op == "Sy")
    {
        a.valid = true;

        a.new_state = local_state ^ 1ULL;

        if(up)
        {
            a.matrix_element =
                Complex(0.0, -0.5);
        }
        else
        {
            a.matrix_element =
                Complex(0.0, 0.5);
        }

        return a;
    }

    throw std::runtime_error(
        "Unknown spin operator: " + op
    );
}

}