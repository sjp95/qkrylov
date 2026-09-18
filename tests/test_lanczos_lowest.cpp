#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <memory>

#include "qkrylov/basis/spinhalf_basis.hpp"
#include "qkrylov/sites/spinhalf_site.hpp"
#include "qkrylov/operators/operator_term.hpp"
#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"
#include "qkrylov/solvers/lanczos.hpp"
#include "qkrylov/solvers/davidson.hpp"
#include "qkrylov/c_api.h"

using namespace qkrylov;
using namespace qkrylov::QKRYLOV_PRECISION_NAMESPACE;

int main()
{
    const int N = 4;
    auto basis = std::make_shared<SpinHalfBasis>(N, Sector(sector::Sz{0}));
    auto site  = std::make_shared<SpinHalfSite>();

    // 4-site AFM Heisenberg ring: J sum_i (S^z_i S^z_{i+1} + 0.5 S^+_i S^-_{i+1} + 0.5 S^-_i S^+_{i+1})
    OpSum ops;
    for (int i = 0; i < N; ++i) {
        int j = (i + 1) % N;
        {
            OperatorTerm t;
            t.coeff = 1.0;
            t.factors = {{"Sz", i}, {"Sz", j}};
            ops.add_term(t);
        }
        {
            OperatorTerm t;
            t.coeff = 0.5;
            t.factors = {{"Sp", i}, {"Sm", j}};
            ops.add_term(t);
        }
        {
            OperatorTerm t;
            t.coeff = 0.5;
            t.factors = {{"Sm", i}, {"Sp", j}};
            ops.add_term(t);
        }
    }

    MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace> H(basis, site, ops);
    const Index dim = H.dimension();
    std::cout << "Hamiltonian dimension: " << dim << std::endl;
    assert(dim == 6);

    // 1. Solve lowest 3 eigenvalues with Davidson for reference
    auto dav_res = davidson_lowest(H, 3, 6, 1e-10);
    std::cout << "Davidson lowest 3 energies:" << std::endl;
    for (size_t k = 0; k < dav_res.eigenvalues.size(); ++k) {
        std::cout << "  E[" << k << "] = " << dav_res.eigenvalues[k] << std::endl;
    }
    assert(dav_res.eigenvalues.size() == 3);

    // 2. Solve lowest 3 eigenvalues with lanczos_lowest
    auto lowest_res = lanczos_lowest(H, 3, 100, 1e-10, true);
    std::cout << "\nLanczos lowest 3 energies:" << std::endl;
    for (size_t k = 0; k < lowest_res.eigenvalues.size(); ++k) {
        std::cout << "  E[" << k << "] = " << lowest_res.eigenvalues[k] << std::endl;
    }
    assert(lowest_res.eigenvalues.size() == 3);
    assert(lowest_res.eigenvectors.size() == 3);

    // Check agreement between Lanczos and Davidson
    for (size_t k = 0; k < 3; ++k) {
        Real diff = std::abs(lowest_res.eigenvalues[k] - dav_res.eigenvalues[k]);
        std::cout << "State " << k << " delta: " << diff << std::endl;
        assert(diff < 1e-6);
    }

    // Check eigenvector orthonormality
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            Complex overlap = 0.0;
            for (Index d = 0; d < dim; ++d) {
                overlap += std::conj(lowest_res.eigenvectors[i][d]) * lowest_res.eigenvectors[j][d];
            }
            Real expected = (i == j) ? 1.0 : 0.0;
            Real err = std::abs(overlap - expected);
            assert(err < 1e-5);
        }
    }
    std::cout << "Eigenvector orthonormality verified." << std::endl;

    // 3. Test warm-starting: pass ground state as initial_vector
    HostVector warm_vec = lowest_res.eigenvectors[0];
    auto warm_res = lanczos_ground_state(H, 100, 1e-10, warm_vec);
    std::cout << "Warm-started ground state energy: " << warm_res.energy << " in " << warm_res.iterations << " iters" << std::endl;
    assert(std::abs(warm_res.energy - lowest_res.eigenvalues[0]) < 1e-6);

    // 4. Test C API
    std::cout << "\n--- Testing C API lanczos_lowest ---" << std::endl;
    qkrylov_sector_h sec = qkrylov_sector_create();
    qkrylov_sector_set_sz(sec, 0);
    qkrylov_basis_h c_basis = qkrylov_spinhalf_basis_create(N, sec);
    qkrylov_site_h  c_site  = qkrylov_spinhalf_site_create();
    qkrylov_opsum_h c_ops   = qkrylov_opsum_create();

    for (int i = 0; i < N; ++i) {
        int j = (i + 1) % N;
        qkrylov_opsum_add_term_2body(c_ops, 1.0, 0.0, "Sz", i, "Sz", j);
        qkrylov_opsum_add_term_2body(c_ops, 0.5, 0.0, "Sp", i, "Sm", j);
        qkrylov_opsum_add_term_2body(c_ops, 0.5, 0.0, "Sm", i, "Sp", j);
    }

    qkrylov_hamiltonian_h c_H = qkrylov_hamiltonian_create(c_basis, c_site, c_ops);
    assert(c_H != nullptr);

    double c_evals[3];
    double c_evecs[3 * 6 * 2]; // 3 vectors, dim 6, complex
    qkrylov_lanczos_lowest_result_c_t c_res_info;

    int rc = qkrylov_lanczos_lowest_complex_fp64(c_H, 3, 100, 1e-10, c_evals, c_evecs, &c_res_info, nullptr);
    assert(rc == QKRYLOV_SUCCESS);
    assert(c_res_info.converged == 1);
    for (int k = 0; k < 3; ++k) {
        assert(std::abs(c_evals[k] - lowest_res.eigenvalues[k]) < 1e-6);
    }
    std::cout << "C API qkrylov_lanczos_lowest_complex_fp64 passed." << std::endl;

    // Test C API with warm start
    int rc_warm = qkrylov_lanczos_lowest_complex_fp64(c_H, 1, 100, 1e-10, c_evals, nullptr, &c_res_info, c_evecs);
    assert(rc_warm == QKRYLOV_SUCCESS);
    assert(std::abs(c_evals[0] - lowest_res.eigenvalues[0]) < 1e-6);
    std::cout << "C API warm-start passed." << std::endl;

    qkrylov_hamiltonian_destroy(c_H);
    qkrylov_opsum_destroy(c_ops);
    qkrylov_site_destroy(c_site);
    qkrylov_basis_destroy(c_basis);
    qkrylov_sector_destroy(sec);

    std::cout << "\nAll test_lanczos_lowest assertions PASSED!" << std::endl;
    return 0;
}
