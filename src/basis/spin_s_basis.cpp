#include "qkrylov/basis/spin_s_basis.hpp"

#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace qkrylov
{

SpinSBasis::SpinSBasis(
    int N,
    double S,
    const Sector& sector
)
    : N_(N),
      S_(S),
      d_(static_cast<int>(std::round(2.0 * S + 1.0))),
      sector_(sector)
{
    if (N <= 0) {
        throw std::runtime_error("SpinSBasis: N must be > 0.");
    }

    if (sector_.use_sz)
        build_sz_basis();
    else
        build_full_basis();
}

Index SpinSBasis::size() const
{
    return states_.size();
}

StateID SpinSBasis::state(Index i) const
{
    return states_.at(i);
}

// Low-RAM optimization: std::lower_bound binary search instead of std::unordered_map!
Index SpinSBasis::index(StateID s) const
{
    auto it = std::lower_bound(states_.begin(), states_.end(), s);
    if (it == states_.end() || *it != s) {
        throw std::runtime_error("State not present in basis");
    }
    return static_cast<Index>(std::distance(states_.begin(), it));
}

bool SpinSBasis::contains(StateID s) const
{
    return std::binary_search(states_.begin(), states_.end(), s);
}

int SpinSBasis::compute_sz2(StateID state) const
{
    StateID temp = state;
    double total_sz = 0.0;
    for (int i = 0; i < N_; ++i) {
        int sigma = static_cast<int>(temp % d_);
        total_sz += (sigma - S_);
        temp /= d_;
    }
    return static_cast<int>(std::round(2.0 * total_sz));
}

void SpinSBasis::build_full_basis()
{
    StateID total_dim = 1;
    for (int i = 0; i < N_; ++i) {
        total_dim *= d_;
    }

    states_.reserve(total_dim);
    for (StateID s = 0; s < total_dim; ++s) {
        states_.push_back(s);
    }
}

void SpinSBasis::build_sz_basis()
{
    StateID total_dim = 1;
    for (int i = 0; i < N_; ++i) {
        total_dim *= d_;
    }

    states_.reserve(total_dim / 2); // heuristic reservation

    for (StateID s = 0; s < total_dim; ++s) {
        if (compute_sz2(s) == sector_.sz2) {
            states_.push_back(s);
        }
    }
    states_.shrink_to_fit();
}

}
