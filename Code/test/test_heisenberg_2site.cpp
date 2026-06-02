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

        t.factors.push_back({"Sz",0});
        t.factors.push_back({"Sz",1});

        os.add_term(t);
    }

    {
        OperatorTerm t;

        t.coeff = 0.5;

        t.factors.push_back({"Sp",0});
        t.factors.push_back({"Sm",1});

        os.add_term(t);
    }

    {
        OperatorTerm t;

        t.coeff = 0.5;

        t.factors.push_back({"Sm",0});
        t.factors.push_back({"Sp",1});

        os.add_term(t);
    }

    MatrixFreeHamiltonian H(
        basis,
        site,
        os
    );

    const Index dim =
        H.dimension();

    std::vector<
    std::vector<Complex>
> M(
    dim,
    std::vector<Complex>(dim)
);

for(Index col=0;
    col<dim;
    ++col)
{
    MatrixFreeHamiltonian::Vector x(
        dim,
        Complex(0.0,0.0)
    );

    x[col] = 1.0;

    MatrixFreeHamiltonian::Vector y;

    H.apply(x,y);

    for(Index row=0;
        row<dim;
        ++row)
    {
        M[row][col] = y[row];
    }
}

std::cout
    << "\nHamiltonian Matrix\n\n";

for(Index row=0;
    row<dim;
    ++row)
{
    for(Index col=0;
        col<dim;
        ++col)
    {
        std::cout
            << M[row][col]
            << " ";
    }

    std::cout
        << "\n";
}
    return 0;
}
