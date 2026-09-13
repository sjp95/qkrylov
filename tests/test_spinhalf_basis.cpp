#include <iostream>

#include "qkrylov/symmetry/sector.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"

using namespace qkrylov;
using namespace qkrylov::QKRYLOV_PRECISION_NAMESPACE;

int main()
{
    basis::SpinHalf basis(10, basis::sector::Sz{0});

    std::cout
        << "Basis dimension (Sz=0) = "
        << basis.size()
        << std::endl;

    if (basis.size() != 252) {
        std::cerr << "Expected dimension 252 for 10 sites Sz=0, got " << basis.size() << std::endl;
        return 1;
    }

    basis::SpinHalf b_full(4, basis::sector::Unconstrained{});
    if (b_full.size() != 16) {
        std::cerr << "Expected dimension 16 for 4 sites unconstrained, got " << b_full.size() << std::endl;
        return 1;
    }

    // Check contains and index roundtrip for Sz=0 basis
    for(Index i = 0; i < basis.size(); ++i) {
        StateID s = basis.state(i);
        if (!basis.contains(s)) {
            std::cerr << "Sz basis should contain state " << s << std::endl;
            return 1;
        }
        if (basis.index(s) != i) {
            std::cerr << "Index mismatch for state " << s << ": expected " << i << ", got " << basis.index(s) << std::endl;
            return 1;
        }
    }

    // State with all ones (1023) has Sz=10, not Sz=0
    if (basis.contains(1023)) {
        std::cerr << "Sz=0 basis should not contain state 1023" << std::endl;
        return 1;
    }

    // Check contains and index for unconstrained basis
    for(Index i = 0; i < b_full.size(); ++i) {
        StateID s = b_full.state(i);
        if (!b_full.contains(s) || b_full.index(s) != i) {
            std::cerr << "Unconstrained basis index mismatch for state " << s << std::endl;
            return 1;
        }
    }
    if (b_full.contains(999)) {
        std::cerr << "Unconstrained basis of 4 sites should not contain state 999" << std::endl;
        return 1;
    }

    return 0;
}