#include <qkrylov/qkrylov.hpp>
#include <iostream>

using namespace qkrylov;

int main() {
    // [ID: init_basis]
    basis::SpinHalf b(10, sector::Sz(0));
    std::cout << "Basis dimension: " << b.dimension() << "\n";
    // [END: init_basis]

    // [ID: build_op]
    OpSum ops;
    double J = 1.0;
    for (int i = 0; i < 9; ++i) {
        ops.add_term(OperatorTerm(J, {{"Sz", i}, {"Sz", i + 1}}));
        ops.add_term(OperatorTerm(J*0.5, {{"Sp", i}, {"Sm", i + 1}}));
        ops.add_term(OperatorTerm(J*0.5, {{"Sm", i}, {"Sp", i + 1}}));
    }
    // [END: build_op]

    // [ID: solve]
    Hamiltonian H(b, ops);
    auto result = lanczos_ground_state(H);
    std::cout << "Ground State Energy: " << result.energy << "\n";
    // [END: solve]
    return 0;
}
