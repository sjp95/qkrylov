#pragma once

#include "basis.hpp"
#include <vector>

namespace qkrylov
{

class GenericBasis : public Basis
{
public:

    GenericBasis(const std::vector<int>& local_dims);

    ~GenericBasis() override = default;

    Index size() const override;

    StateID state(Index i) const override;

    Index index(StateID s) const override;

    bool contains(StateID s) const override;

    int nsites() const noexcept override;

private:

    void build_basis();

private:

    std::vector<int> local_dims_;

    std::vector<StateID> states_;
};

} // namespace qkrylov
