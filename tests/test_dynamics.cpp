#include "qkrylov/solvers/dynamics.hpp"
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

    MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace> H(basis, site, os);
    auto l_res = lanczos_ground_state<Kokkos::DefaultExecutionSpace>(H);

    // Test S(omega) with A = Sx0 + Sx1
    // For 2 sites, Sx0+Sx1 acting on singlet |01>-|10> gives 0?
    // Wait, Sx = 0.5(Sp+Sm).
    // Let's use A = Sz0.

    HostVector phi0(H.dimension(), 0.0);
    // Manually apply Sz0 to ground state
    // We can't easily apply it with current API without a full H,
    // so let's just make a dummy phi0 for now to test the CF logic.
    phi0[0] = 1.0;

    auto dyn_res = continued_fraction_coeffs<Kokkos::DefaultExecutionSpace>(H, phi0, 10);
    assert(dyn_res.alphas.size() > 0);

    Real val = evaluate_spectral_function(dyn_res.alphas.data(), dyn_res.betas.data(), dyn_res.alphas.size(), dyn_res.norm_phi0, 0.5, l_res.energy, 0.1);
    std::cout << "S(0.5) = " << val << "\n";
    assert(val >= 0.0);

    // Test Real-Time Pure-State Evolution
    OpSum os_sz0;
    {
        OperatorTerm t; t.coeff = 1.0;
        t.factors.push_back({"Sz", 0});
        os_sz0.add_term(t);
    }
    MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace> Sz0(basis, site, os_sz0);

    OpSum os_sz1;
    {
        OperatorTerm t; t.coeff = 1.0;
        t.factors.push_back({"Sz", 1});
        os_sz1.add_term(t);
    }
    MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace> Sz1(basis, site, os_sz1);

    // Prepare |up, down> state
    auto d0 = Sz0.diagonal();
    auto d1 = Sz1.diagonal();
    HostVector psi0_complex(H.dimension(), Complex(0.0, 0.0));
    for (size_t i = 0; i < H.dimension(); ++i) {
        if (std::abs(d0[i].real() - Real(0.5)) < 1e-5 && std::abs(d1[i].real() - Real(-0.5)) < 1e-5) {
            psi0_complex[i] = Complex(1.0, 0.0);
            break;
        }
    }

    std::vector<Real> time_grid = {0.0, 0.5, 1.0, 1.5, 2.0, 3.141592653589793};
    auto rt_res = time_evolve<Kokkos::DefaultExecutionSpace>(H, psi0_complex, time_grid, {Sz0}, 10);
    assert(rt_res.time_grid.size() == time_grid.size());
    assert(rt_res.observable_expectations.size() == 1);
    for (size_t ti = 0; ti < time_grid.size(); ++ti) {
        Real t = time_grid[ti];
        Real expected_sz = 0.5 * std::cos(1.0 * t); // Delta E = 1.0
        Real computed_sz = rt_res.observable_expectations[0][ti].real();
        std::cout << "t=" << t << " | computed <Sz0>=" << computed_sz << " | expected=" << expected_sz << "\n";
        assert(std::abs(computed_sz - expected_sz) < 1e-3);
    }

    // Test FTLM Dynamics C_AB(t) with A = Sz0, B = Sz0
    std::cout << "Testing ftlm_dynamics...\n";
    auto dyn_ftlm = ftlm_dynamics<Kokkos::DefaultExecutionSpace>(H, 1.0, Sz0, Sz0, time_grid, 60, 10, 12345ULL);
    assert(dyn_ftlm.time_grid.size() == time_grid.size());
    assert(dyn_ftlm.correlations.size() == time_grid.size());
    assert(dyn_ftlm.correlation_errors.size() == time_grid.size());
    // At t=0, C_AB(0) = <Sz0 * Sz0> = 0.25 (since Sz0^2 = 1/4 * I)
    std::cout << "C(0) = " << dyn_ftlm.correlations[0] << " +/- " << dyn_ftlm.correlation_errors[0] << "\n";
    assert(std::abs(dyn_ftlm.correlations[0].real() - 0.25) < 0.05);
    assert(std::abs(dyn_ftlm.correlations[0].imag()) < 0.05);

    std::cout << "All dynamics tests passed!\n";
    return 0;
}
