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

    // Heisenberg model for 2 sites: H = J * (Sz1*Sz2 + 0.5*(Sp1*Sm2 + Sm1*Sp2))
    // For J=1, ground state is singlet with E = -0.75

    {
        OperatorTerm t;
        t.coeff = 1.0;
        t.factors.push_back({"Sz", 0});
        t.factors.push_back({"Sz", 1});
        os.add_term(t);
    }
    {
        OperatorTerm t;
        t.coeff = 0.5;
        t.factors.push_back({"Sp", 0});
        t.factors.push_back({"Sm", 1});
        os.add_term(t);
    }
    {
        OperatorTerm t;
        t.coeff = 0.5;
        t.factors.push_back({"Sm", 0});
        t.factors.push_back({"Sp", 1});
        os.add_term(t);
    }

    MatrixFreeHamiltonian H(basis, site, os);
    auto res = lanczos_ground_state(H);

    std::cout << "Lanczos Energy: " << res.energy << " (Expected -0.75)\n";
    assert(std::abs(res.energy + 0.75) < 1e-10);

    // Verify Ritz vector: H * v should be energy * v
    Vector Hv(H.dimension());
    H.apply(res.eigenvector, Hv);

    for (Index i = 0; i < H.dimension(); ++i) {
        Complex diff = Hv[i] - res.energy * res.eigenvector[i];
        assert(std::abs(diff) < 1e-10);
    }

    std::cout << "Ritz vector verification passed!\n";

    return 0;
}
