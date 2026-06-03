#include <iostream>

#include "qkrylov/sites/spinhalf_site.hpp"

using namespace qkrylov;

int main()
{
    SpinHalfSite site;

    StateID state = 0;

    auto sx =
        site.apply(
            "Sx",
            0,
            state
        );

    std::cout
        << "Sx:\n";

    std::cout
        << "valid = "
        << sx.valid
        << "\n";

    std::cout
        << "new_state = "
        << sx.new_state
        << "\n";

    std::cout
        << "matrix_element = "
        << sx.matrix_element
        << "\n\n";

    auto sy =
        site.apply(
            "Sy",
            0,
            state
        );

    std::cout
        << "Sy:\n";

    std::cout
        << "valid = "
        << sy.valid
        << "\n";

    std::cout
        << "new_state = "
        << sy.new_state
        << "\n";

    std::cout
        << "matrix_element = "
        << sy.matrix_element
        << "\n\n";

    auto sz =
        site.apply(
            "Sz",
            0,
            state
        );

    std::cout
        << "Sz:\n";

    std::cout
        << "valid = "
        << sz.valid
        << "\n";

    std::cout
        << "new_state = "
        << sz.new_state
        << "\n";

    std::cout
        << "matrix_element = "
        << sz.matrix_element
        << "\n";

        auto sp =
    site.apply(
        "Sp",
        0,
        state
    );

    std::cout
        << "\nSp:\n";

    std::cout
        << "valid = "
        << sp.valid
        << "\n";

    std::cout
        << "new_state = "
        << sp.new_state
        << "\n";

    std::cout
        << "matrix_element = "
        << sp.matrix_element
        << "\n";

    StateID up_state = 1;

auto sm =
    site.apply(
        "Sm",
        0,
        up_state
    );

    std::cout
        << "\nSm:\n";

    std::cout
        << "valid = "
        << sm.valid
        << "\n";

    std::cout
        << "new_state = "
        << sm.new_state
        << "\n";

    std::cout
        << "matrix_element = "
        << sm.matrix_element
        << "\n";

    return 0;
}