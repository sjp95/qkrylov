#include "qkrylov/sites/fermion_site.hpp"
#include <iostream>
#include <cassert>

using namespace qkrylov;

int main() {
    FermionSite site;
    StateID s = 0b1010; // occupied at 1 and 3

    // Test N on occupied site
    auto aN = site.apply("N", 1ULL);
    assert(aN.valid && aN.matrix_element == 1.0);

    // Test N on empty site
    aN = site.apply("N", 0ULL);
    assert(aN.valid && aN.matrix_element == 0.0);

    // Test C on occupied site
    auto aC = site.apply("C", 1ULL);
    assert(aC.valid && aC.matrix_element == 1.0 && aC.new_state == 0ULL);

    std::cout << "FermionSite tests passed!\n";
    return 0;
}
