#include "qkrylov/solvers/ftlm.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"
#include "qkrylov/sites/spinhalf_site.hpp"
#include "qkrylov/operators/opsum.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace qkrylov;
using namespace qkrylov::QKRYLOV_PRECISION_NAMESPACE;

int main() {
    // -------------------------------------------------------------------------
    // Test 1: 2-site H = Sz0 * Sz1 baseline test
    // -------------------------------------------------------------------------
    {
        int N = 2;
        auto basis = std::make_shared<SpinHalfBasis>(N);
        auto site = std::make_shared<SpinHalfSite>();
        OpSum os;
        {
            OperatorTerm t; t.coeff = 1.0;
            t.factors.push_back({"Sz", 0}); t.factors.push_back({"Sz", 1});
            os.add_term(t);
        }
        MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace> H(basis, site, os);

        Real beta = 1.0;
        auto res = ftlm<Kokkos::DefaultExecutionSpace>(H, beta, 100, 10);

        Real z_val = res.partition_function(beta);
        Real e_val = res.internal_energy(beta);

        std::cout << "2-site FTLM Partition Function Z: " << z_val << "\n";
        std::cout << "2-site FTLM Internal Energy E: " << e_val << "\n";

        assert(z_val > 0);
        assert(std::abs(e_val) < 1.0);
    }

    // -------------------------------------------------------------------------
    // Test 2: 4-site Heisenberg chain <Mz^2>_beta vs Exact Values
    // -------------------------------------------------------------------------
    {
        int N = 4;
        auto basis = std::make_shared<SpinHalfBasis>(N);
        auto site = std::make_shared<SpinHalfSite>();

        // H = sum_{i=0}^{3} (Sz_i Sz_{i+1} + 0.5 Sp_i Sm_{i+1} + 0.5 Sm_i Sp_{i+1}) with periodic boundary
        OpSum os_H;
        OpSum os_Mz;
        for (int i = 0; i < N; ++i) {
            int j = (i + 1) % N;
            { OperatorTerm t; t.coeff = 1.0; t.factors.push_back({"Sz", i}); t.factors.push_back({"Sz", j}); os_H.add_term(t); }
            { OperatorTerm t; t.coeff = 0.5; t.factors.push_back({"Sp", i}); t.factors.push_back({"Sm", j}); os_H.add_term(t); }
            { OperatorTerm t; t.coeff = 0.5; t.factors.push_back({"Sm", i}); t.factors.push_back({"Sp", j}); os_H.add_term(t); }

            { OperatorTerm t; t.coeff = 1.0; t.factors.push_back({"Sz", i}); os_Mz.add_term(t); }
        }

        // Mz^2 operator: (sum_i Sz_i)^2 = sum_i Sz_i^2 + 2 sum_{i<j} Sz_i Sz_j
        OpSum os_Mz2;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                OperatorTerm t; t.coeff = 1.0;
                t.factors.push_back({"Sz", i});
                t.factors.push_back({"Sz", j});
                os_Mz2.add_term(t);
            }
        }

        MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace> H(basis, site, os_H);
        MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace> Mz2(basis, site, os_Mz2);

        // Run FTLM sampling
        int n_random = 200;
        int n_steps = 16;
        auto ftlm_res = ftlm<Kokkos::DefaultExecutionSpace>(H, 1.0, n_random, n_steps);

        std::vector<Real> betas = {0.1, 0.5, 1.0, 2.0};
        // Reference exact values for 4-site periodic Heisenberg chain <Mz^2>_beta:
        // beta=0.1: ~0.950042
        // beta=0.5: ~0.755081
        // beta=1.0: ~0.537883
        // beta=2.0: ~0.238406
        std::vector<Real> exact_mz2 = {0.950042, 0.755081, 0.537883, 0.238406};

        std::vector<Real> ftlm_mz2 = ftlm_res.expectation_value(Mz2, betas);

        std::cout << "\n4-site Heisenberg <Mz^2>_beta FTLM vs Exact:\n";
        for (size_t k = 0; k < betas.size(); ++k) {
            Real err = std::abs(ftlm_mz2[k] - exact_mz2[k]) / exact_mz2[k];
            std::cout << "beta = " << betas[k]
                      << ": FTLM = " << ftlm_mz2[k]
                      << ", Exact = " << exact_mz2[k]
                      << ", Relative Error = " << err * 100.0 << "%\n" << std::flush;

            // Verify <Mz^2> matches exact diagonalization within stochastic tolerance (10%)
            assert(err < 0.05);
        }
    }

    std::cout << "\nAll FTLM C++ tests passed successfully!\n";
    return 0;
}
