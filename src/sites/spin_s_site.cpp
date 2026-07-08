#include "qkrylov/sites/spin_s_site.hpp"
#include <cmath>
#include <stdexcept>

namespace qkrylov
{

SpinSSite::SpinSSite(double S) : S_(S), dim_((int)(2*S + 1)) {}

LocalAction SpinSSite::apply(const std::string& op, StateID local_state) const
{
    // encoding: m value from -S to S maps to index i = m + S (0 to 2S)
    int idx = (int)local_state;
    double m = (double)idx - S_;

    LocalAction a;
    if (op == "Sz") {
        a.valid = true;
        a.new_state = local_state;
        a.matrix_element = m;
    } else if (op == "Sp") {
        if (idx >= dim_ - 1) return a;
        a.valid = true;
        a.new_state = local_state + 1ULL;
        a.matrix_element = std::sqrt(S_*(S_+1.0) - m*(m+1.0));
    } else if (op == "Sm") {
        if (idx <= 0) return a;
        a.valid = true;
        a.new_state = local_state - 1ULL;
        a.matrix_element = std::sqrt(S_*(S_+1.0) - m*(m-1.0));
    } else {
        throw std::runtime_error("Unknown Spin-S operator: " + op);
    }
    return a;
}

}
