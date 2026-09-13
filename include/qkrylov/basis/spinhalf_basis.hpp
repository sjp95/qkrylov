#pragma once

#include "qkrylov/core/types.hpp"

#include "basis.hpp"
#include "../symmetry/sector.hpp"

#include <vector>
#include <memory>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {


class SpinHalfBasis : public Basis
{
public:

    SpinHalfBasis(
        int N,
        const Sector& sector = Sector{}
    );

    SpinHalfBasis(
        int N,
        const sector::Sz& sz
    ) : SpinHalfBasis(N, Sector(sz)) {}

    SpinHalfBasis(
        int N,
        const sector::Unconstrained& u
    ) : SpinHalfBasis(N, Sector(u)) {}

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
};

} // namespace QKRYLOV_PRECISION_NAMESPACE

namespace basis {
    using SpinHalf = QKRYLOV_PRECISION_NAMESPACE::SpinHalfBasis;
    namespace sector = qkrylov::sector;
}

} // namespace qkrylov