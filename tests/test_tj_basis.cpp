#include "qkrylov/basis/tj_basis.hpp"
#include <iostream>
#include <cassert>

using namespace qkrylov;

int main() {
    // 2 sites, no symmetry.
    // Possible states per site: empty, up, down. (3^2 = 9)
    // Hubbard has 4^2 = 16.
    TJBasis b1(2);
    std::cout << "TJ Basis size (2 sites): " << b1.size() << " (Expected 9)\n";
    assert(b1.size() == 9);

    return 0;
}
