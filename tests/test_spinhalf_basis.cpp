#include <iostream>

#include "qkrylov/symmetry/sector.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"

using namespace qkrylov;

int main()
{
    Sector sec;

    sec.use_sz = true;
    sec.sz2 = 0;

    SpinHalfBasis basis(10, sec);

    std::cout
        << "Basis dimension = "
        << basis.size()
        << std::endl;

    for(Index i = 0; i < 10 && i < basis.size(); ++i)
    {
        std::cout
            << i
            << "  "
            << basis.state(i)
            << '\n';
    }

    return 0;
}