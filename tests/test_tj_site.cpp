#include "qkrylov/sites/tj_site.hpp"
#include <iostream>
#include <cassert>

using namespace qkrylov;

int main() {
    TJSite site;
    StateID s = 0b0001; // site 0 has Up electron

    // Test CUp
    auto a = site.apply("CUp", s);
    assert(a.valid && a.new_state == 0 && a.matrix_element == 1.0);

    // Test CdagDn - should be invalid because Up is already there
    auto b = site.apply("CdagDn", s);
    assert(!b.valid);

    std::cout << "TJSite tests passed!\n";
    return 0;
}
