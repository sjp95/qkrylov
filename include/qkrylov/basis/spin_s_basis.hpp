#pragma once

#include "basis.hpp"
#include "../symmetry/sector.hpp"

#include <vector>
#include <algorithm>
#include <memory>
#include <cmath>

namespace qkrylov
{

class SpinSBasis : public Basis
{
public:

    SpinSBasis(
        int N,
        double S = 0.5,
        const Sector& sector = Sector{}
    );

    ~SpinSBasis() override = default;

    Index size() const override;

    StateID state(Index i) const override;

    Index index(StateID s) const override;

    bool contains(StateID s) const override;

    int nsites() const noexcept { return N_; }
    double spin() const noexcept { return S_; }
    int dimension_per_site() const noexcept { return d_; }

    const Sector& sector() const noexcept { return sector_; }

private:

    void build_full_basis();

    void build_sz_basis();

    int compute_sz2(StateID state) const;

private:

    int N_;
    double S_;
    int d_;
    Sector sector_;

    std::vector<StateID> states_;
};

}
