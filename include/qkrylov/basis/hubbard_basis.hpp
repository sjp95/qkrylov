#pragma once

#include "basis.hpp"
#include "../symmetry/sector.hpp"

#include <vector>
#include <unordered_map>
#include <memory>

namespace qkrylov
{

class HubbardBasis : public Basis
{
public:

    HubbardBasis(
        int N,
        const Sector& sector = Sector{}
    );

    ~HubbardBasis() override = default;

    Index size() const override;

    StateID state(Index i) const override;

    Index index(StateID s) const override;

    bool contains(StateID s) const override;

    int nsites() const noexcept override
    {
        return N_;
    }

    const Sector& sector() const noexcept
    {
        return sector_;
    }

private:

    void build_full_basis();

    void build_nup_ndn_basis();

private:

    int N_;

    Sector sector_;

    std::vector<StateID> states_;

    std::unordered_map<StateID, Index> lookup_;
};

} // namespace qkrylov
