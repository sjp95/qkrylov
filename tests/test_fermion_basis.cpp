#include "qkrylov/basis/fermion_basis.hpp"
#include <iostream>
#include <cassert>

using namespace qkrylov;

int main() {
    // Test full basis
    FermionBasis b1(4);
    std::cout << "Full basis size: " << b1.size() << " (Expected 16)\n";
    assert(b1.size() == 16);

    // Test n-conserved basis
    Sector sec;
    sec.use_n = true;
    sec.n = 2;
    FermionBasis b2(4, sec);
    std::cout << "N=2 basis size: " << b2.size() << " (Expected 6)\n";
    assert(b2.size() == 6);

    for (Index i = 0; i < b2.size(); ++i) {
        StateID s = b2.state(i);
        std::cout << "State " << i << ": " << s << "\n";
    }

    return 0;
}
