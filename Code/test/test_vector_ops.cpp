#include <iostream>

#include "qkrylov/linalg/vector_ops.hpp"

using namespace qkrylov;

int main()
{
    Vector x(3);
    Vector y(3);

    x[0]=1.0;
    x[1]=2.0;
    x[2]=3.0;

    y[0]=4.0;
    y[1]=5.0;
    y[2]=6.0;

    std::cout
        << "dot = "
        << dot(x,y)
        << "\n";

    std::cout
        << "norm(x) = "
        << norm(x)
        << "\n";

    return 0;
}
