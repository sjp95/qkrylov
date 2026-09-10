#include "qkrylov/solvers/lanczos.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"
#include "qkrylov/sites/spinhalf_site.hpp"
#include "qkrylov/operators/opsum.hpp"
#include <iostream>
#include <cassert>

using namespace qkrylov;
using namespace qkrylov::QKRYLOV_PRECISION_NAMESPACE;

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

    MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace> H(basis, site, os);

    // Test default / two-pass
    auto res_tp = lanczos_ground_state<Kokkos::DefaultExecutionSpace>(H, 200, 1e-12, true);
    std::cout << "Two-Pass Lanczos Energy: " << res_tp.energy << " (Expected -0.75)\n";
    assert(std::abs(res_tp.energy + 0.75) < 1e-5);

    // Test single-pass
    auto res_sp = lanczos_ground_state<Kokkos::DefaultExecutionSpace>(H, 200, 1e-12, false);
    std::cout << "Single-Pass Lanczos Energy: " << res_sp.energy << " (Expected -0.75)\n";
    assert(std::abs(res_sp.energy + 0.75) < 1e-5);

    // Verify Ritz vector: H * v should be energy * v for two-pass
    HostVector Hv(H.dimension());
    H.apply(res_tp.eigenvector.data(), Hv.data());

    for (Index i = 0; i < H.dimension(); ++i) {
        Complex diff = Hv[i] - Complex(res_tp.energy * res_tp.eigenvector[i].real(), res_tp.energy * res_tp.eigenvector[i].imag());
        assert(std::abs(diff) < 1e-5);
    }

    // Verify two-pass and single-pass match
    for (Index i = 0; i < H.dimension(); ++i) {
        Complex diff = res_tp.eigenvector[i] - res_sp.eigenvector[i];
        assert(std::abs(diff) < 1e-5);
    }

    std::cout << "Ritz vector verification passed for two-pass and single-pass!\n";

    return 0;
}
