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
    auto basis =
        std::make_shared<basis::SpinHalf>(
            2,
            basis::sector::Unconstrained{}
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

    LanczosConfig config;
    config.maxiter = 200;
    config.tol = (sizeof(Real) == 4) ? Real(1e-6) : Real(1e-12);

    auto res_op = solvers::lanczos<solvers::policy::OnePass>(H, config);
    auto res_dkgs = solvers::lanczos<solvers::policy::OnePass_DKGS>(H, config);
    auto res_full = solvers::lanczos<solvers::policy::OnePass_full>(H, config);
    auto res_tp = solvers::lanczos<solvers::policy::TwoPass>(H, config);

    // Verify OnePass: energy only, eigenvector is empty
    assert(res_op.eigenvector.empty());

    // Verify structured binding unpack
    const Real bind_tol = (sizeof(Real) == 4) ? Real(1e-5) : Real(1e-10);
    auto [e, v] = solvers::lanczos<solvers::policy::TwoPass>(H, config);
    if (std::abs(e - res_tp.energy) > bind_tol) {
        std::cerr << "Structured binding energy mismatch!\n";
        return 1;
    }
    if (v.size() != res_tp.eigenvector.size()) {
        std::cerr << "Structured binding eigenvector size mismatch!\n";
        return 1;
    }

    std::cout << "Energy (OnePass)     = " << res_op.energy << "\n";
    std::cout << "Energy (OnePass_DKGS)= " << res_dkgs.energy << "\n";
    std::cout << "Energy (OnePass_full)= " << res_full.energy << "\n";
    std::cout << "Energy (TwoPass)     = " << res_tp.energy << "\n";
    std::cout << "Energy (structured)  = " << e << "\n";

    const Real comp_tol = (sizeof(Real) == 4) ? Real(1e-4) : Real(1e-10);
    if (std::abs(res_dkgs.energy - res_tp.energy) > comp_tol ||
        std::abs(res_op.energy - res_tp.energy) > comp_tol ||
        std::abs(res_full.energy - res_tp.energy) > comp_tol) {
        std::cerr << "Mismatch between Lanczos policy energies!\n";
        return 1;
    }

    return 0;
}
