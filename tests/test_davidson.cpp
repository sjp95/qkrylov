#include "qkrylov/solvers/davidson.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"
#include "qkrylov/sites/spinhalf_site.hpp"
#include "qkrylov/operators/opsum.hpp"
#include <iostream>

using namespace qkrylov;

int main() {
    int N = 4;
    auto basis = std::make_shared<SpinHalfBasis>(N);
    auto site = std::make_shared<SpinHalfSite>();
    OpSum os;
    for (int i = 0; i < N; ++i) {
        OperatorTerm t;
        t.coeff = 1.0;
        t.factors.push_back({"Sz", i});
        os.add_term(t);
    }
    // H = sum Sz_i. Ground state should be all spins down, energy -N/2 = -2.0

    MatrixFreeHamiltonian H(basis, site, os);
    auto res = davidson_lowest(H, 1);

    std::cout << "Davidson Energy: " << res.eigenvalues[0] << " (Expected -2.0)\n";

    return 0;
}
