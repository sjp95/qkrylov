#include "qkrylov/solvers/lanczos.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"
#include "qkrylov/sites/spinhalf_site.hpp"
#include "qkrylov/operators/opsum.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <memory>
#include <stdexcept>

using namespace qkrylov;
using namespace qkrylov::QKRYLOV_PRECISION_NAMESPACE;

int main() {
    std::cout << "=== Running test_lanczos_policies ===" << std::endl;

    const int N = 4;
    auto basis = std::make_shared<SpinHalfBasis>(N, Sector(sector::Sz{0}));
    auto site  = std::make_shared<SpinHalfSite>();

    // 4-site AFM Heisenberg ring
    OpSum ops;
    for (int i = 0; i < N; ++i) {
        int j = (i + 1) % N;
        OperatorTerm tz(1.0, {{"Sz", i}, {"Sz", j}});
        OperatorTerm txy1(0.5, {{"Sp", i}, {"Sm", j}});
        OperatorTerm txy2(0.5, {{"Sm", i}, {"Sp", j}});
        ops.add_term(tz);
        ops.add_term(txy1);
        ops.add_term(txy2);
    }

    MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace> H(basis, site, ops);
    const Index dim = H.dimension();
    assert(dim == 6);

    const Real tol = (sizeof(Real) == 4) ? Real(1e-5) : Real(1e-10);
    const Real comp_tol = (sizeof(Real) == 4) ? Real(1e-4) : Real(1e-8);
    const Real expected_E0 = Real(-2.0);

    // -------------------------------------------------------------------------
    // 1. Test policy::OnePass (Ground State Energy Only)
    // -------------------------------------------------------------------------
    std::cout << "--- 1. Testing policy::OnePass (Ground state energy only) ---" << std::endl;
    {
        LanczosConfig cfg;
        cfg.maxiter = 100;
        cfg.tol = tol;

        auto res = solvers::lanczos<solvers::policy::OnePass>(H, cfg);
        std::cout << "OnePass E0: " << res.energy << ", iters: " << res.iterations << std::endl;
        assert(std::abs(res.energy - expected_E0) < comp_tol);
        assert(res.eigenvector.empty()); // No vector allocation
        assert(res.converged);

        // Scope check: n_eig > 1 must throw
        bool threw = false;
        try {
            LanczosConfig bad_cfg = cfg;
            bad_cfg.n_eig = 2;
            solvers::lanczos<solvers::policy::OnePass>(H, bad_cfg);
        } catch (const std::invalid_argument& e) {
            threw = true;
            std::cout << "OnePass multi-state rejection: " << e.what() << std::endl;
        }
        assert(threw);
    }

    // -------------------------------------------------------------------------
    // 2. Test policy::OnePass_DKGS (Arbitrary Number of Low Energy States)
    // -------------------------------------------------------------------------
    std::cout << "\n--- 2. Testing policy::OnePass_DKGS (Arbitrary low energy states) ---" << std::endl;
    LanczosResult res_dkgs;
    {
        LanczosConfig cfg;
        cfg.n_eig = 3;
        cfg.maxiter = 100;
        cfg.tol = tol;

        res_dkgs = solvers::lanczos<solvers::policy::OnePass_DKGS>(H, cfg);
        std::cout << "OnePass_DKGS energies:" << std::endl;
        for (size_t k = 0; k < res_dkgs.eigenvalues.size(); ++k) {
            std::cout << "  E[" << k << "] = " << res_dkgs.eigenvalues[k] << std::endl;
        }

        assert(res_dkgs.eigenvalues.size() == 3);
        assert(res_dkgs.eigenvectors.size() == 3);
        assert(!res_dkgs.eigenvector.empty());
        assert(std::abs(res_dkgs.energy - expected_E0) < comp_tol);
        assert(std::abs(res_dkgs.eigenvalues[0] - expected_E0) < comp_tol);

        // Verify Ritz residuals: ||H v - E v|| < comp_tol
        for (size_t k = 0; k < 3; ++k) {
            HostVector Hv(dim);
            H.apply(res_dkgs.eigenvectors[k].data(), Hv.data());
            Real res_norm = 0.0;
            for (Index d = 0; d < dim; ++d) {
                Complex diff = Hv[d] - Complex(res_dkgs.eigenvalues[k] * res_dkgs.eigenvectors[k][d].real(),
                                               res_dkgs.eigenvalues[k] * res_dkgs.eigenvectors[k][d].imag());
                res_norm += std::norm(diff);
            }
            res_norm = std::sqrt(res_norm);
            std::cout << "  Ritz residual state " << k << ": " << res_norm << std::endl;
            assert(res_norm < comp_tol);
        }

        // Verify DGKS mutual basis vector / Ritz vector orthogonality
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                Complex ov = 0.0;
                for (Index d = 0; d < dim; ++d) {
                    ov += std::conj(res_dkgs.eigenvectors[i][d]) * res_dkgs.eigenvectors[j][d];
                }
                Real exp_ov = (i == j) ? 1.0 : 0.0;
                assert(std::abs(ov - exp_ov) < comp_tol);
            }
        }
        std::cout << "OnePass_DKGS orthogonality verified." << std::endl;
    }

    // -------------------------------------------------------------------------
    // 3. Test policy::OnePass_full (Ground State Only with Vector Return)
    // -------------------------------------------------------------------------
    std::cout << "\n--- 3. Testing policy::OnePass_full (Ground state only + vector) ---" << std::endl;
    {
        LanczosConfig cfg;
        cfg.maxiter = 100;
        cfg.tol = tol;

        auto res = solvers::lanczos<solvers::policy::OnePass_full>(H, cfg);
        std::cout << "OnePass_full E0: " << res.energy << ", iters: " << res.iterations << std::endl;
        assert(std::abs(res.energy - expected_E0) < comp_tol);
        assert(!res.eigenvector.empty());
        assert(res.eigenvectors.size() == 1);

        // Check Ritz residual
        HostVector Hv(dim);
        H.apply(res.eigenvector.data(), Hv.data());
        Real res_norm = 0.0;
        for (Index d = 0; d < dim; ++d) {
            Complex diff = Hv[d] - Complex(res.energy * res.eigenvector[d].real(),
                                           res.energy * res.eigenvector[d].imag());
            res_norm += std::norm(diff);
        }
        assert(std::sqrt(res_norm) < comp_tol);

        // Scope check: n_eig > 1 must throw
        bool threw = false;
        try {
            LanczosConfig bad_cfg = cfg;
            bad_cfg.n_eig = 2;
            solvers::lanczos<solvers::policy::OnePass_full>(H, bad_cfg);
        } catch (const std::invalid_argument& e) {
            threw = true;
            std::cout << "OnePass_full multi-state rejection: " << e.what() << std::endl;
        }
        assert(threw);
    }

    // -------------------------------------------------------------------------
    // 4. Test policy::TwoPass (Ground State Only with Vector Return, Replay)
    // -------------------------------------------------------------------------
    std::cout << "\n--- 4. Testing policy::TwoPass (Ground state only + vector, replay) ---" << std::endl;
    {
        LanczosConfig cfg;
        cfg.maxiter = 100;
        cfg.tol = tol;

        auto res = solvers::lanczos<solvers::policy::TwoPass>(H, cfg);
        std::cout << "TwoPass E0: " << res.energy << ", iters: " << res.iterations << std::endl;
        assert(std::abs(res.energy - expected_E0) < comp_tol);
        assert(!res.eigenvector.empty());
        assert(res.eigenvectors.size() == 1);

        // Check overlap with OnePass_DKGS ground state: |<v_TP | v_DKGS>| == 1
        Complex ov = 0.0;
        for (Index d = 0; d < dim; ++d) {
            ov += std::conj(res.eigenvector[d]) * res_dkgs.eigenvector[d];
        }
        std::cout << "TwoPass overlap with OnePass_DKGS: " << std::abs(ov) << std::endl;
        assert(std::abs(std::abs(ov) - 1.0) < comp_tol);

        // Test warm-starting replay in TwoPass
        cfg.initial_vector = res.eigenvector;
        auto res_warm = solvers::lanczos<solvers::policy::TwoPass>(H, cfg);
        assert(std::abs(res_warm.energy - expected_E0) < comp_tol);
        std::cout << "TwoPass warm-start converged in " << res_warm.iterations << " iters" << std::endl;

        // Scope check: n_eig > 1 must throw
        bool threw = false;
        try {
            LanczosConfig bad_cfg = cfg;
            bad_cfg.n_eig = 2;
            solvers::lanczos<solvers::policy::TwoPass>(H, bad_cfg);
        } catch (const std::invalid_argument& e) {
            threw = true;
            std::cout << "TwoPass multi-state rejection: " << e.what() << std::endl;
        }
        assert(threw);
    }

    // -------------------------------------------------------------------------
    // 5. Test Structured Bindings Unpack Across All Policies
    // -------------------------------------------------------------------------
    std::cout << "\n--- 5. Testing Structured Bindings ---" << std::endl;
    {
        auto [e1, v1] = solvers::lanczos<solvers::policy::OnePass>(H);
        assert(std::abs(e1 - expected_E0) < comp_tol);
        assert(v1.empty());

        auto [e2, v2] = solvers::lanczos<solvers::policy::OnePass_DKGS>(H);
        assert(std::abs(e2 - expected_E0) < comp_tol);
        assert(!v2.empty());

        auto [e3, v3] = solvers::lanczos<solvers::policy::OnePass_full>(H);
        assert(std::abs(e3 - expected_E0) < comp_tol);
        assert(!v3.empty());

        auto [e4, v4] = solvers::lanczos<solvers::policy::TwoPass>(H);
        assert(std::abs(e4 - expected_E0) < comp_tol);
        assert(!v4.empty());

        std::cout << "Structured bindings verified." << std::endl;
    }

    std::cout << "\nAll test_lanczos_policies checks PASSED!" << std::endl;
    return 0;
}
