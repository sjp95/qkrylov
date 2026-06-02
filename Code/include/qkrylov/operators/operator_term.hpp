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
};

}