#include "qkrylov/sites/spin_s_site.hpp"
#include "qkrylov/basis/spin_s_basis.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"
#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/solvers/lanczos.hpp"
#include "qkrylov/solvers/correction_vector.hpp"
#include <cassert>
#include <iostream>

using namespace qkrylov;
using namespace qkrylov::QKRYLOV_PRECISION_NAMESPACE;

int main() {
    std::cout << "=== Running Test: Spin-1 & Correction Vector Spectroscopy ===" << std::endl;

    // 1. Setup a 4-site Spin-1 Heisenberg Chain (S = 1.0)
    const int N = 4;
    const double S = 1.0;

    auto site = std::make_shared<SpinSSite>(S);

    // Sz = 0 sector (dim = 19)
    Sector sector;
    sector.use_sz = true;
    sector.sz2 = 0; // Total Sz = 0

    auto basis = std::make_shared<SpinSBasis>(N, S, sector);
    std::cout << "Spin-1 (N=" << N << ", Sz=0) basis dimension: " << basis->size() << std::endl;
    assert(basis->size() == 19);

    // H = \sum_{i=0}^2 [ S^z_i S^z_{i+1} + 0.5 (S^+_i S^-_{i+1} + S^-_i S^+_{i+1}) ]
    OpSum ops;
    for (int i = 0; i < N - 1; ++i) {
        OperatorTerm t_sz;
        t_sz.coeff = 1.0;
        t_sz.factors.push_back({"Sz", i});
        t_sz.factors.push_back({"Sz", i + 1});
        ops.add_term(t_sz);

        OperatorTerm t_sp;
        t_sp.coeff = 0.5;
        t_sp.factors.push_back({"Sp", i});
        t_sp.factors.push_back({"Sm", i + 1});
        ops.add_term(t_sp);

        OperatorTerm t_sm;
        t_sm.coeff = 0.5;
        t_sm.factors.push_back({"Sm", i});
        t_sm.factors.push_back({"Sp", i + 1});
        ops.add_term(t_sm);
    }

    MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace> H(basis, site, ops);
    assert(H.dimension() == 19);

    // 2. Solve Ground State using Lanczos
    auto lgs = lanczos_ground_state<Kokkos::DefaultExecutionSpace>(H);
    std::cout << "Lanczos ground state energy: " << lgs.energy << std::endl;
    assert(lgs.converged);
    assert(lgs.eigenvector.size() == 19);

    // 3. Apply excitation operator S^z_0 to ground state
    OpSum op_sz0;
    OperatorTerm t0;
    t0.coeff = 1.0;
    t0.factors.push_back({"Sz", 0});
    op_sz0.add_term(t0);

    MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace> H_sz0(basis, site, op_sz0);
    HostVector Op_psi0(H.dimension(), Complex(0.0, 0.0));
    H_sz0.apply(lgs.eigenvector.data(), Op_psi0.data());

    // 4. Correction vector spectroscopy solve at omega = 1.5, eta = 0.1
    const Real omega = 1.5;
    const Real eta = 0.1;
    auto cv_res = correction_vector_spectral<Kokkos::DefaultExecutionSpace>(
        H, Op_psi0, lgs.energy, omega, eta
    );

    std::cout << "Correction Vector Solver Converged: " << (cv_res.converged ? "YES" : "NO")
              << " in " << cv_res.iterations << " iterations." << std::endl;
    std::cout << "Spectral function S(omega=" << omega << "): " << cv_res.spectral_function << std::endl;

    assert(cv_res.converged);
    assert(cv_res.spectral_function >= 0.0);
    assert(cv_res.correction_vector.size() == 19);

    std::cout << "=== Spin-1 & Correction Vector Test PASSED ===" << std::endl;
    return 0;
}
