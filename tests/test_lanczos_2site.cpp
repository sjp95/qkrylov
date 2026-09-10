#include <iostream>
#include <memory>
#include <cassert>
#include <cmath>

#include "qkrylov/symmetry/sector.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"
#include "qkrylov/operators/operator_term.hpp"
#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/sites/spinhalf_site.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"
#include "qkrylov/solvers/lanczos.hpp"

using namespace qkrylov;

int main()
{
    Sector sec;

    auto basis = std::make_shared<SpinHalfBasis>(2, sec);
    auto site = std::make_shared<SpinHalfSite>();

    OpSum os;

    {
        OperatorTerm t;
        t.coeff = 1.0;
        t.factors.push_back({"Sz",0});
        t.factors.push_back({"Sz",1});
        os.add_term(t);
    }

    {
        OperatorTerm t;
        t.coeff = 0.5;
        t.factors.push_back({"Sp",0});
        t.factors.push_back({"Sm",1});
        os.add_term(t);
    }

    {
        OperatorTerm t;
        t.coeff = 0.5;
        t.factors.push_back({"Sm",0});
        t.factors.push_back({"Sp",1});
        os.add_term(t);
    }

    MatrixFreeHamiltonian H(basis, site, os);

    // Test Two-Pass Lanczos (two_pass = true)
    auto res_tp = lanczos_ground_state(H, 200, 1e-12, true);
    std::cout << "Two-Pass Energy = " << res_tp.energy << "\n";

    // Test Single-Pass Lanczos (two_pass = false)
    auto res_sp = lanczos_ground_state(H, 200, 1e-12, false);
    std::cout << "Single-Pass Energy = " << res_sp.energy << "\n";

    // Verify energies match exact singlet energy (-0.75) and each other
    assert(std::abs(res_tp.energy - (-0.75)) < 1e-6);
    assert(std::abs(res_sp.energy - (-0.75)) < 1e-6);
    assert(std::abs(res_tp.energy - res_sp.energy) < 1e-10);

    // Verify eigenvectors match
    assert(res_tp.eigenvector.size() == res_sp.eigenvector.size());
    for (size_t i = 0; i < res_tp.eigenvector.size(); ++i) {
        assert(std::abs(res_tp.eigenvector[i] - res_sp.eigenvector[i]) < 1e-6);
    }

    std::cout << "Lanczos two-pass and single-pass tests passed!\n";
    return 0;
}
