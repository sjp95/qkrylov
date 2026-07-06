#include <qkrylov/basis/spinhalf_basis.hpp>
#include <qkrylov/operators/opsum.hpp>
#include <qkrylov/sites/spinhalf_site.hpp>
#include <qkrylov/hamiltonian/matrix_free_hamiltonian.hpp>
#include <qkrylov/solvers/lanczos.hpp>
#include <iostream>

using namespace qkrylov;

int main() {
    int N = 4;
    auto basis = std::make_shared<SpinHalfBasis>(N);
    auto site = std::make_shared<SpinHalfSite>();

    OpSum os;
    for (int i = 0; i < N - 1; ++i) {
        // Heisenberg interaction: Sz_i Sz_{i+1} + 0.5(Sp_i Sm_{i+1} + Sm_i Sp_{i+1})
        os += {1.0, {{"Sz", i}, {"Sz", i+1}}};
        os += {0.5, {{"Sp", i}, {"Sm", i+1}}};
        os += {0.5, {{"Sm", i}, {"Sp", i+1}}};
    }

    MatrixFreeHamiltonian H(basis, site, os);
    auto result = lanczos_ground_state(H);

    std::cout << "Ground state energy: " << result.energy << std::endl;
    return 0;
}
