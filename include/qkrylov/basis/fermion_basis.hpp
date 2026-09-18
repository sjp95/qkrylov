#pragma once

#include "qkrylov/core/types.hpp"

#include "basis.hpp"
#include "../symmetry/sector.hpp"

#include <vector>
#include <memory>

namespace qkrylov {

class FermionBasis : public Basis
{
public:

    FermionBasis(
        int N,
        const Sector& sector = Sector{}
    );

    FermionBasis(
        int N,
        const sector::Particles& p
    ) : FermionBasis(N, Sector(p)) {}

    FermionBasis(
        int N,
        const sector::Unconstrained& u
    ) : FermionBasis(N, Sector(u)) {}

    ~FermionBasis() override = default;

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

    void build_n_basis();

private:

    int N_;

    Sector sector_;

    std::vector<StateID> states_;
};

namespace QKRYLOV_PRECISION_NAMESPACE {
using qkrylov::FermionBasis;
}

namespace basis {
    using Fermion = qkrylov::FermionBasis;
    namespace sector = qkrylov::sector;
}

} // namespace qkrylov
