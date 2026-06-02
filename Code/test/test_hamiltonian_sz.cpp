#include <iostream>
#include <memory>

#include "qkrylov/symmetry/sector.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"

#include "qkrylov/operators/operator_term.hpp"
#include "qkrylov/operators/opsum.hpp"

#include "qkrylov/sites/spinhalf_site.hpp"

#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"

using namespace qkrylov;

int main()
{
    Sector sec;

    SpinHalfBasis basis(
        1,
        sec
    );

    auto basis_ptr =
        std::make_shared<SpinHalfBasis>(
            basis
        );

    auto site_ptr =
        std::make_shared<SpinHalfSite>();

    OpSum os;

    OperatorTerm term;

    term.coeff = 1.0;

    term.factors.push_back(
        {"Sz",0}
    );

    os.add_term(term);

    MatrixFreeHamiltonian H(
        basis_ptr,
        site_ptr,
        os
    );

    MatrixFreeHamiltonian::Vector x(2);

    x[0] = 1.0;
    x[1] = 1.0;

    MatrixFreeHamiltonian::Vector y;

    H.apply(
        x,
        y
    );

    std::cout
        << "y[0] = "
        << y[0]
        << "\n";

    std::cout
        << "y[1] = "
        << y[1]
        << "\n";

    return 0;
}
