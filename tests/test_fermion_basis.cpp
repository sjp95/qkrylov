#include "qkrylov/basis/fermion_basis.hpp"
#include <iostream>
#include <cassert>

using namespace qkrylov;
using namespace qkrylov::QKRYLOV_PRECISION_NAMESPACE;

int main() {
    // Test full basis
    basis::Fermion b1(4, basis::sector::Unconstrained{});
    std::cout << "Full basis size: " << b1.size() << " (Expected 16)\n";
    assert(b1.size() == 16);

    // Test n-conserved basis
    basis::Fermion b2(4, basis::sector::Particles{2});
    std::cout << "N=2 basis size: " << b2.size() << " (Expected 6)\n";
    assert(b2.size() == 6);

    for (Index i = 0; i < b2.size(); ++i) {
        StateID s = b2.state(i);
        std::cout << "State " << i << ": " << s << "\n";
    }

    return 0;
}
