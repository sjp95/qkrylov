#include "qkrylov/sites/spin_s_site.hpp"
#include "qkrylov/basis/spin_s_basis.hpp"
#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"
#include "qkrylov/solvers/lanczos.hpp"
#include "qkrylov/solvers/correction_vector.hpp"

#include <iostream>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "=== Running Test: Spin-1 & Correction Vector Spectroscopy ===\n";

    // 1. Setup a 4-site Spin-1 Heisenberg Chain (S = 1.0)
    const int N = 4;
    const double S = 1.0;

    auto site = std::make_shared<qkrylov::SpinSSite>(S);

    // Sz = 0 sector
    qkrylov::Sector sector;
    sector.use_sz = true;
    sector.sz2 = 0; // Total Sz = 0

    auto basis = std::make_shared<qkrylov::SpinSBasis>(N, S, sector);
    std::cout << "Spin-1 (N=" << N << ", Sz=0) basis dimension: " << basis->size() << "\n";

    qkrylov::OpSum ops;
    for (int i = 0; i < N - 1; ++i) {
        // H = \sum J (Sz_i Sz_{i+1} + 0.5 Sp_i Sm_{i+1} + 0.5 Sm_i Sp_{i+1})
        ops += qkrylov::OperatorTerm{1.0, {{"Sz", i}, {"Sz", i+1}}};
        ops += qkrylov::OperatorTerm{0.5, {{"Sp", i}, {"Sm", i+1}}};
        ops += qkrylov::OperatorTerm{0.5, {{"Sm", i}, {"Sp", i+1}}};
    }

    qkrylov::MatrixFreeHamiltonian H(basis, site, ops);

    // 2. Solve Ground State using Lanczos
    auto lanczos_res = qkrylov::lanczos_ground_state(H);
    std::cout << "Ground state energy E0: " << lanczos_res.energy << "\n";

    // 3. Apply operator O = Sz_0 to ground state
    qkrylov::OpSum op_sz0;
    op_sz0 += qkrylov::OperatorTerm{1.0, {{"Sz", 0}}};
    qkrylov::MatrixFreeHamiltonian H_sz0(basis, site, op_sz0);

    qkrylov::Vector Op_psi0(H.dimension(), 0.0);
    H_sz0.apply(lanczos_res.eigenvector.data(), Op_psi0.data());

    // 4. Run Correction Vector Spectroscopy at target energy omega = 1.5
    double omega = 1.5;
    double eta = 0.1;
    auto cv_res = qkrylov::correction_vector_spectral(H, Op_psi0, lanczos_res.energy, omega, eta);

    std::cout << "Correction Vector Solver Converged: " << (cv_res.converged ? "YES" : "NO")
              << " in " << cv_res.iterations << " iterations.\n";
    std::cout << "Spectral function S(omega=" << omega << "): " << cv_res.spectral_function << "\n";

    assert(cv_res.converged);
    assert(cv_res.spectral_function >= 0.0);

    std::cout << "=== Spin-1 & Correction Vector Test PASSED ===\n";
    return 0;
}
