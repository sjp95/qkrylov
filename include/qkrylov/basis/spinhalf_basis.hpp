#pragma once

#include "basis.hpp"
#include "../symmetry/sector.hpp"

#include <vector>
#include <unordered_map>
#include <memory>

namespace qkrylov
{

class SpinHalfBasis : public Basis
{
public:

    SpinHalfBasis(
        int N,
        const Sector& sector = Sector{}
    );

    ~SpinHalfBasis() override = default;

    Index size() const override;

    StateID state(Index i) const override;

    Index index(StateID s) const override;

    bool contains(StateID s) const override;

    int nsites() const noexcept
    {
        return N_;
    }

    const Sector& sector() const noexcept
    {
        return sector_;
    }

private:

    void build_full_basis();

    void build_sz_basis();

    static int compute_sz2(
        StateID state,
        int N
    );

private:

    int N_;

    Sector sector_;

    std::vector<StateID> states_;

    std::unordered_map<StateID, Index> lookup_;
};

} // namespace qkrylov