#pragma once

#include "qkrylov/operators/local_action.hpp"

#include <string>

namespace qkrylov
{

class Site
{
public:

    virtual ~Site() = default;

    virtual LocalAction apply(
        const std::string& op,
        int site,
        StateID state
    ) const = 0;
};

}