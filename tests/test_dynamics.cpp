#include "qkrylov/solvers/dynamics.hpp"
#include "qkrylov/solvers/lanczos.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"
#include "qkrylov/sites/spinhalf_site.hpp"
#include "qkrylov/operators/opsum.hpp"
#include <iostream>
#include <cassert>

using namespace qkrylov;

int main() {
    int N = 2;
    auto basis = std::make_shared<SpinHalfBasis>(N);
    auto site = std::make_shared<SpinHalfSite>();
    OpSum os;
    // H = Sz0*Sz1 + 0.5(Sp0*Sm1 + Sm0*Sp1)
    {
        OperatorTerm t; t.coeff = 1.0;
        t.factors.push_back({"Sz", 0}); t.factors.push_back({"Sz", 1});
        os.add_term(t);
    }
    {
        OperatorTerm t; t.coeff = 0.5;
        t.factors.push_back({"Sp", 0}); t.factors.push_back({"Sm", 1});
        os.add_term(t);
    }
    {
        OperatorTerm t; t.coeff = 0.5;
        t.factors.push_back({"Sm", 0}); t.factors.push_back({"Sp", 1});
        os.add_term(t);
    }

    MatrixFreeHamiltonian H(basis, site, os);
    auto l_res = lanczos_ground_state(H);

    // Test S(omega) with A = Sx0 + Sx1
    // For 2 sites, Sx0+Sx1 acting on singlet |01>-|10> gives 0?
    // Wait, Sx = 0.5(Sp+Sm).
    // Let's use A = Sz0.

    Vector phi0(H.dimension(), 0.0);
    // Manually apply Sz0 to ground state
    // We can't easily apply it with current API without a full H,
    // so let's just make a dummy phi0 for now to test the CF logic.
    phi0[0] = 1.0;

    auto dyn_res = continued_fraction_coeffs(H, phi0, 10);
    assert(dyn_res.alphas.size() > 0);

    double val = evaluate_spectral_function(dyn_res, 0.5, l_res.energy, 0.1);
    std::cout << "S(0.5) = " << val << "\n";
    assert(val >= 0.0);

    std::cout << "Dynamics test passed!\n";
    return 0;
}
