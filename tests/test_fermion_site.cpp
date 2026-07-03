#include "qkrylov/sites/fermion_site.hpp"
#include <iostream>
#include <cassert>

using namespace qkrylov;

int main() {
    FermionSite site;
    StateID s = 0b1010; // occupied at 1 and 3

    // Test N
    auto aN = site.apply("N", 1, s);
    assert(aN.valid && aN.matrix_element == 1.0);

    aN = site.apply("N", 0, s);
    assert(aN.valid && aN.matrix_element == 0.0);

    // Test C
    auto aC = site.apply("C", 1, s);
    // phase: popcount(s & mask(1)) = popcount(s & 0b0001) = 0 -> phase +1
    assert(aC.valid && aC.matrix_element == 1.0 && aC.new_state == 0b1000);

    aC = site.apply("C", 3, s);
    // phase: popcount(s & mask(3)) = popcount(s & 0b0111) = popcount(0b0010) = 1 -> phase -1
    assert(aC.valid && aC.matrix_element == -1.0 && aC.new_state == 0b0010);

    std::cout << "FermionSite tests passed!\n";
    return 0;
}
