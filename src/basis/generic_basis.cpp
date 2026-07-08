#include "qkrylov/basis/generic_basis.hpp"
#include <stdexcept>
#include <algorithm>

namespace qkrylov
{

GenericBasis::GenericBasis(const std::vector<int>& local_dims)
    : local_dims_(local_dims)
{
    build_basis();
}

Index GenericBasis::size() const
{
    return states_.size();
}

StateID GenericBasis::state(Index i) const
{
    return states_.at(i);
}

Index GenericBasis::index(StateID s) const
{
    auto it = std::lower_bound(states_.begin(), states_.end(), s);
    if (it == states_.end() || *it != s)
        throw std::runtime_error("State not present in basis");
    return std::distance(states_.begin(), it);
}

bool GenericBasis::contains(StateID s) const
{
    return std::binary_search(states_.begin(), states_.end(), s);
}

int GenericBasis::nsites() const noexcept
{
    return (int)local_dims_.size();
}

void GenericBasis::build_basis()
{
    Index total_dim = 1;
    for (int d : local_dims_) total_dim *= d;

    states_.reserve(total_dim);
    for (StateID s = 0; s < (StateID)total_dim; ++s) {
        states_.push_back(s);
    }
}

} // namespace qkrylov
