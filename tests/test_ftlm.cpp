#include "qkrylov/solvers/ftlm.hpp"
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
    // H = Sz0*Sz1
    {
        OperatorTerm t; t.coeff = 1.0;
        t.factors.push_back({"Sz", 0}); t.factors.push_back({"Sz", 1});
        os.add_term(t);
    }
    MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace> H(basis, site, os);

    Real beta = 1.0;
    auto res = ftlm<Kokkos::DefaultExecutionSpace>(H, beta, 100, 10);

    std::cout << "FTLM Partition Function Z: " << res.partition_function << "\n";
    std::cout << "FTLM Internal Energy E: " << res.internal_energy << "\n";

    // For 2 sites with H=Sz0*Sz1:
    // Energies: 1/4, -1/4, -1/4, 1/4
    // Z = 2*exp(-beta/4) + 2*exp(beta/4) = 4*cosh(beta/4)
    // For beta=1, Z = 4*cosh(0.25) approx 4*1.0314 = 4.125

    assert(res.partition_function > 0);
    assert(std::abs(res.internal_energy) < 1.0);

    // Test low-temperature stability at beta = 50.0 (where exp(-beta * E0) without shift would overflow float)
    Real beta_low = 50.0;
    auto res_low = ftlm<Kokkos::DefaultExecutionSpace>(H, beta_low, 50, 10);
    std::cout << "Low-T FTLM (beta=50) Internal Energy E: " << res_low.internal_energy << "\n";
    std::cout << "Low-T FTLM (beta=50) Specific Heat Cv: " << res_low.specific_heat << "\n";

    assert(!std::isnan(res_low.internal_energy));
    assert(!std::isnan(res_low.specific_heat));
    assert(std::abs(res_low.internal_energy - (-0.25)) < 0.1);
    assert(res_low.specific_heat >= 0.0);

    std::cout << "FTLM test passed!\n";
    return 0;
}
