#pragma once

#include "qkrylov/core/types.hpp"
#include <cmath>

namespace qkrylov {

namespace sector {

// Unconstrained Hilbert space (no symmetries applied)
struct Unconstrained {};

// Total Sz conservation (supports integer or half-integer)
struct Sz {
    int sz2 = 0; // Stores 2 * Sz internally to represent half-integers cleanly
    constexpr Sz() = default;
    constexpr explicit Sz(int val) : sz2(val * 2) {}
    constexpr explicit Sz(double val) : sz2(static_cast<int>(std::round(val * 2.0))) {}
};

// Fermionic particle number conservation
struct Particles {
    int n = 0;
    constexpr Particles() = default;
    constexpr explicit Particles(int count) : n(count) {}
};
using ParticleNumber = Particles;

// Hubbard / t-J spin-resolved particle conservation
struct Hubbard {
    int nup = 0;
    int ndn = 0;
    constexpr Hubbard() = default;
    constexpr Hubbard(int up, int dn) : nup(up), ndn(dn) {}
};

// Bosonic particle conservation
struct Bosons {
    int nb = 0;
    constexpr Bosons() = default;
    constexpr explicit Bosons(int count) : nb(count) {}
};

} // namespace sector

struct Sector
{
    //
    // Spin quantum number conservation
    //
    bool use_sz = false;

    // 2*Sz
    int sz2 = 0;

    //
    // Future Hubbard/t-J support
    //
    bool use_nup = false;
    bool use_ndn = false;

    int nup = 0;
    int ndn = 0;

    //
    // Fermion particle number conservation
    //
    bool use_n = false;
    int n = 0;

    //
    // Future boson support
    //
    bool use_nb = false;

    int nb = 0;

    Sector() = default;
    Sector(const sector::Unconstrained&) {}
    Sector(const sector::Sz& s) : use_sz(true), sz2(s.sz2) {}
    Sector(const sector::Particles& p) : use_n(true), n(p.n) {}
    Sector(const sector::Hubbard& h) : use_nup(true), use_ndn(true), nup(h.nup), ndn(h.ndn) {}
    Sector(const sector::Bosons& b) : use_nb(true), nb(b.nb) {}
};

namespace QKRYLOV_PRECISION_NAMESPACE {

namespace sector = qkrylov::sector;
using qkrylov::Sector;

} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov