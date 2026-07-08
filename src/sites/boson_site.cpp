#include "qkrylov/sites/boson_site.hpp"
#include <cmath>
#include <stdexcept>

namespace qkrylov
{

BosonSite::BosonSite(int max_occupancy) : max_occ_(max_occupancy) {}

LocalAction BosonSite::apply(const std::string& op, StateID local_state) const
{
    int n = (int)local_state;

    LocalAction a;
    if (op == "N") {
        a.valid = true;
        a.new_state = local_state;
        a.matrix_element = (double)n;
    } else if (op == "B") {
        if (n == 0) return a;
        a.valid = true;
        a.new_state = local_state - 1ULL;
        a.matrix_element = std::sqrt((double)n);
    } else if (op == "Bdag") {
        if (n >= max_occ_) return a;
        a.valid = true;
        a.new_state = local_state + 1ULL;
        a.matrix_element = std::sqrt((double)n + 1.0);
    } else {
        throw std::runtime_error("Unknown boson operator: " + op);
    }
    return a;
}

}
