#undef NDEBUG
#include <cassert>
#include <iostream>
#include <memory>
#include <cmath>

#include "qkrylov/symmetry/sector.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"
#include "qkrylov/operators/operator_term.hpp"
#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/operators/local_op.hpp"
#include "qkrylov/sites/spinhalf_site.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"
#include "qkrylov/solvers/lanczos.hpp"

using namespace qkrylov;
using namespace qkrylov::QKRYLOV_PRECISION_NAMESPACE;

void test_heisenberg_workflow() {
    std::cout << "Running test_heisenberg_workflow..." << std::endl;

    // 4-site chain with Sz=0 sector (dimension = C(4,2) = 6)
    auto basis = basis::SpinHalf(4, basis::sector::Sz{0});
    assert(basis.size() == 6 && "Sz=0 sector of 4-site chain should have dim 6");
    assert(basis.nsites() == 4 && "Basis should have 4 sites");

    // Use the algebraic expression template syntax
    OpSum os;
    for (int i = 0; i < 3; ++i) {
        os += 1.0 * Sz(i) * Sz(i+1) + 0.5 * Sp(i) * Sm(i+1) + 0.5 * Sm(i) * Sp(i+1);
    }
    assert(os.size() == 9 && "Should have 9 terms for 3 bonds");

    Hamiltonian H(basis, os, device::cpu{});
    assert(H.dimension() == 6 && "Hamiltonian dimension should match Sz=0 sector size");

    LanczosConfig cfg;
    cfg.maxiter = 200;
    cfg.tol = 1e-12;
    auto [res_energy, res_vec] = solvers::lanczos<solvers::policy::OnePass_DKGS>(H, cfg);

    // Exact ground state energy for 4-site Heisenberg chain (OBC) = 1 - sqrt(2) ≈ -0.6160254038
    // but restricted to Sz=0 sector the ground state is still -1.6160254038
    Real exact_energy = -1.6160254038;
    std::cout << "Computed Energy: " << res_energy << " Exact: " << exact_energy << std::endl;
    assert(std::abs(res_energy - exact_energy) < 1e-5 && "Ground state energy should match exact 4-site Heisenberg value");
    assert(res_vec.size() == 6 && "Eigenvector dimension should match Sz=0 sector size");

    std::cout << "test_heisenberg_workflow PASSED!" << std::endl;
}

int main() {
    test_heisenberg_workflow();
    std::cout << "All workflow tests passed successfully." << std::endl;
    return 0;
}
