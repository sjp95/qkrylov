#include <iostream>
#include <memory>

#include "qkrylov/symmetry/sector.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"

#include "qkrylov/operators/operator_term.hpp"
#include "qkrylov/operators/opsum.hpp"

#include "qkrylov/sites/spinhalf_site.hpp"

#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"

#include "qkrylov/solvers/lanczos.hpp"

using namespace qkrylov;
using namespace qkrylov::QKRYLOV_PRECISION_NAMESPACE;

int main()
{
    Sector sec;

    auto basis =
        std::make_shared<SpinHalfBasis>(
            2,
            sec
        );

    auto site =
        std::make_shared<SpinHalfSite>();

    OpSum os;

    {
        OperatorTerm t;

        t.coeff = 1.0;

        t.factors.push_back(
            {"Sz",0}
        );

        t.factors.push_back(
            {"Sz",1}
        );

        os.add_term(t);
    }

    {
        OperatorTerm t;

        t.coeff = 0.5;

        t.factors.push_back(
            {"Sp",0}
        );

        t.factors.push_back(
            {"Sm",1}
        );

        os.add_term(t);
    }

    {
        OperatorTerm t;

        t.coeff = 0.5;

        t.factors.push_back(
            {"Sm",0}
        );

        t.factors.push_back(
            {"Sp",1}
        );

        os.add_term(t);
    }

    MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace> H(
        basis,
        site,
        os
    );

    auto res_two_pass =
        lanczos_ground_state<Kokkos::DefaultExecutionSpace>(
            H,
            200,
            1e-12,
            true
        );

    auto res_single_pass =
        lanczos_ground_state<Kokkos::DefaultExecutionSpace>(
            H,
            200,
            1e-12,
            false
        );

    std::cout
        << "Two-pass energy = "
        << res_two_pass.energy
        << ", Single-pass energy = "
        << res_single_pass.energy
        << "\n";

    assert(std::abs(res_two_pass.energy - (-0.75)) < 1e-10);
    assert(std::abs(res_single_pass.energy - (-0.75)) < 1e-10);
    assert(std::abs(res_two_pass.energy - res_single_pass.energy) < 1e-12);

    for (size_t i = 0; i < res_two_pass.eigenvector.size(); ++i) {
        Complex diff = res_two_pass.eigenvector[i] - res_single_pass.eigenvector[i];
        assert(std::abs(diff) < 1e-10);
    }

    return 0;
}
