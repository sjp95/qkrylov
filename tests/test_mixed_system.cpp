#include "qkrylov/basis/generic_basis.hpp"
#include "qkrylov/sites/fermion_site.hpp"
#include "qkrylov/sites/boson_site.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"
#include "qkrylov/solvers/lanczos.hpp"
#include <iostream>
#include <memory>

using namespace qkrylov;

int main() {
    // Mixed system: Fermion (dim 2) and Phonon (dim 3)
    std::vector<int> local_dims = {2, 3};
    auto basis = std::make_shared<GenericBasis>(local_dims);

    std::vector<std::shared_ptr<Site>> sites = {
        std::make_shared<FermionSite>(),
        std::make_shared<BosonSite>(2)
    };

    OpSum os;
    // Electron-phonon coupling: g * n_0 * (b_1 + bdag_1)
    double g = 0.2;
    os += {g, {{"N", 0}, {"B", 1}}};
    os += {g, {{"N", 0}, {"Bdag", 1}}};

    // Phonon energy: omega * bdag_1 * b_1
    os += {1.0, {{"N", 1}}};

    MatrixFreeHamiltonian H(basis, sites, os);
    auto res = lanczos_ground_state(H);

    std::cout << "Mixed system energy: " << res.energy << "\n";
    return 0;
}
