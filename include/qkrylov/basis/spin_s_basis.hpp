#pragma once

#include "qkrylov/core/types.hpp"
#include "qkrylov/basis/basis.hpp"
#include "qkrylov/symmetry/sector.hpp"

#include <vector>
#include <algorithm>
#include <memory>
#include <cmath>

namespace qkrylov {

class SpinSBasis : public Basis
{
public:

    SpinSBasis(
        int N,
        double S = 0.5,
        const Sector& sector = Sector{}
    );

    SpinSBasis(int N, double S, const sector::Sz& sz)
        : SpinSBasis(N, S, Sector(sz)) {}

    SpinSBasis(int N, double S, const sector::Unconstrained& u)
        : SpinSBasis(N, S, Sector(u)) {}

    SpinSBasis(int N, const sector::Sz& sz)
        : SpinSBasis(N, 0.5, Sector(sz)) {}

    SpinSBasis(int N, const sector::Unconstrained& u)
        : SpinSBasis(N, 0.5, Sector(u)) {}

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

namespace QKRYLOV_PRECISION_NAMESPACE {
using qkrylov::SpinSBasis;
}

namespace basis {
    using SpinS = qkrylov::SpinSBasis;
    namespace sector = qkrylov::sector;
}

} // namespace qkrylov
