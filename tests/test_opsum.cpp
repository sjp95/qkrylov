#include <iostream>

#include "qkrylov/operators/opsum.hpp"

using namespace qkrylov;

int main()
{
    OpSum os;

    OperatorTerm t;

    t.coeff = 1.0;

    t.factors.push_back(
        {"Sz",0}
    );

    t.factors.push_back(
        {"Sz",1}
    );

    os.add_term(t);

    std::cout
        << os.size()
        << std::endl;

    return 0;
}