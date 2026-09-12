#include "qkrylov/solvers/ftlm.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"
#include "qkrylov/sites/spinhalf_site.hpp"
#include "qkrylov/operators/opsum.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace qkrylov;
using namespace qkrylov::QKRYLOV_PRECISION_NAMESPACE;

int main() {
    std::cout << "--- Testing FTLM Decoupled Multi-Temperature Sweep & Observables ---\n";

    const int N = 2;
    auto basis = std::make_shared<SpinHalfBasis>(N);
    auto site = std::make_shared<SpinHalfSite>();

    // 1. Hamiltonian: H = J * (Sz0*Sz1 + 0.5*(S+0*S-1 + S-0*S+1)) with J = 1.0
    OpSum os_H;
    {
        OperatorTerm t; t.coeff = 1.0;
        t.factors.push_back({"Sz", 0}); t.factors.push_back({"Sz", 1});
        os_H.add_term(t);
    }
    {
        OperatorTerm t; t.coeff = 0.5;
        t.factors.push_back({"Sp", 0}); t.factors.push_back({"Sm", 1});
        os_H.add_term(t);
    }
    {
        OperatorTerm t; t.coeff = 0.5;
        t.factors.push_back({"Sm", 0}); t.factors.push_back({"Sp", 1});
        os_H.add_term(t);
    }
    MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace> H(basis, site, os_H);

    // 2. Observable 1: H itself (to verify <O_H> == <H> algebraically)
    MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace> O_H(basis, site, os_H);

    // 3. Observable 2: Spin-spin correlation Sz0 * Sz1
    OpSum os_szsz;
    {
        OperatorTerm t; t.coeff = 1.0;
        t.factors.push_back({"Sz", 0}); t.factors.push_back({"Sz", 1});
        os_szsz.add_term(t);
    }
    MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace> O_SzSz(basis, site, os_szsz);

    std::vector<MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace>> observables = {O_H, O_SzSz};

    // 4. Test Decoupled Stage 1 Sampling
    const int n_random = 60;
    const int n_steps = 10;
    const uint64_t seed = 12345ULL;

    std::cout << "Running Stage 1: ftlm_sample with " << n_random << " random vectors...\n";
    auto samples = ftlm_sample<Kokkos::DefaultExecutionSpace>(H, observables, n_random, n_steps, seed);
    assert(samples.size() == static_cast<size_t>(n_random));
    for (const auto& s : samples) {
        assert(s.eigenvalues.size() > 0);
        assert(s.projected_operators.size() == 2);
        assert(s.projected_operators[0].matrix.size() == s.eigenvalues.size() * s.eigenvalues.size());
    }
    std::cout << "Stage 1 completed successfully!\n";

    // 5. Test Stage 2: Instant Multi-Temperature Sweep
    std::vector<Real> beta_grid = {0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0};
    std::cout << "Running Stage 2: ftlm_evaluate_sweep over " << beta_grid.size() << " temperatures...\n";
    auto sweep = ftlm_evaluate_sweep(samples, beta_grid);

    assert(sweep.beta_grid.size() == beta_grid.size());
    assert(sweep.partition_functions.size() == beta_grid.size());
    assert(sweep.internal_energies.size() == beta_grid.size());
    assert(sweep.free_energies.size() == beta_grid.size());
    assert(sweep.specific_heats.size() == beta_grid.size());
    assert(sweep.entropies.size() == beta_grid.size());
    assert(sweep.observable_expectations.size() == 2);
    assert(sweep.observable_errors.size() == 2);

    for (size_t bi = 0; bi < beta_grid.size(); ++bi) {
        Real beta = beta_grid[bi];
        Real Z = sweep.partition_functions[bi];
        Real E = sweep.internal_energies[bi];
        Real F = sweep.free_energies[bi];
        Real Cv = sweep.specific_heats[bi];
        Real S = sweep.entropies[bi];
        Real exp_H = sweep.observable_expectations[0][bi];
        Real exp_SzSz = sweep.observable_expectations[1][bi];
        Real err_SzSz = sweep.observable_errors[1][bi];

        std::cout << "beta=" << beta
                  << " | Z=" << Z
                  << " | E=" << E
                  << " | F=" << F
                  << " | Cv=" << Cv
                  << " | S=" << S
                  << " | <O_H>=" << exp_H
                  << " | <SzSz>=" << exp_SzSz << " +/- " << err_SzSz
                  << "\n";

        // Invariants & sanity checks:
        assert(Z > 0.0);
        assert(!std::isnan(E));
        assert(!std::isnan(Cv));
        assert(!std::isnan(S));
        assert(Cv >= Real(0.0)); // Heat capacity must be non-negative
        assert(S >= Real(0.0));  // Entropy must be non-negative

        // Observable <O_H> must match <H> to high precision
        assert(std::abs(exp_H - E) < 1e-4);

        // Ground-state checks at large beta (singlet E0 = -0.75, <SzSz> = -0.25)
        if (beta >= 20.0) {
            assert(std::abs(E - (-0.75)) < 0.05);
            assert(std::abs(exp_SzSz - (-0.25)) < 0.05);
        }

        // High-temperature limit checks at small beta (<SzSz> -> 0)
        if (beta <= 0.1) {
            assert(std::abs(exp_SzSz) < 0.05);
        }
    }

    // 6. Test Zero-Cost Re-evaluation with different grid on same samples
    std::vector<Real> fine_grid = {0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.6, 1.8};
    auto sweep_fine = ftlm_evaluate_sweep(samples, fine_grid);
    assert(sweep_fine.internal_energies.size() == fine_grid.size());

    // 7. Test Convenience Wrapper ftlm_sweep
    auto sweep_direct = ftlm_sweep<Kokkos::DefaultExecutionSpace>(H, {1.0, 2.0}, observables, 20, 10, seed);
    assert(sweep_direct.partition_functions.size() == 2);
    assert(sweep_direct.observable_expectations.size() == 2);

    std::cout << "All FTLM sweep and observable tests PASSED successfully!\n";
    return 0;
}
