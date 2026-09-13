#pragma once

#include "qkrylov/core/types.hpp"

#include "basis.hpp"
#include "../symmetry/sector.hpp"

#include <vector>
#include <memory>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {


class HubbardBasis : public Basis
{
public:

    HubbardBasis(
        int N,
        const Sector& sector = Sector{}
    );

    HubbardBasis(
        int N,
        const sector::Hubbard& h
    ) : HubbardBasis(N, Sector(h)) {}

    HubbardBasis(
        int N,
        const sector::Unconstrained& u
    ) : HubbardBasis(N, Sector(u)) {}

    ~HubbardBasis() override = default;

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

    void build_nup_ndn_basis();

private:

    int N_;

    Sector sector_;

    std::vector<StateID> states_;
};

} // namespace QKRYLOV_PRECISION_NAMESPACE

namespace basis {
    using Hubbard = QKRYLOV_PRECISION_NAMESPACE::HubbardBasis;
    namespace sector = qkrylov::sector;
}

} // namespace qkrylov
