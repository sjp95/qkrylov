#pragma once

#include "qkrylov/core/types.hpp"

#include <string>
#include <vector>

namespace qkrylov
{

struct OperatorFactor
{
    std::string op;
    int site;
};

struct OperatorTerm
{
    Complex coeff;

    std::vector<OperatorFactor> factors;

    OperatorTerm() = default;

    OperatorTerm(Complex c, std::initializer_list<OperatorFactor> f)
        : coeff(c), factors(f) {}
};

}