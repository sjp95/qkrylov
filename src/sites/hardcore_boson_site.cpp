#include "qkrylov/sites/hardcore_boson_site.hpp"
#include <stdexcept>

namespace qkrylov
{

LocalAction HardcoreBosonSite::apply(const std::string& op, StateID local_state) const
{
    bool occupied = local_state & 1ULL;
    LocalAction a;
    if (op == "N") {
        a.valid = true;
        a.new_state = local_state;
        a.matrix_element = occupied ? 1.0 : 0.0;
    } else if (op == "B") {
        if (!occupied) return a;
        a.valid = true;
        a.new_state = 0ULL;
        a.matrix_element = 1.0;
    } else if (op == "Bdag") {
        if (occupied) return a;
        a.valid = true;
        a.new_state = 1ULL;
        a.matrix_element = 1.0;
    } else {
        throw std::runtime_error("Unknown hardcore boson operator: " + op);
    }
    return a;
}

}
