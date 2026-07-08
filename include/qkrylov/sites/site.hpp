#pragma once

#include "qkrylov/operators/local_action.hpp"

#include <string>

namespace qkrylov
{

class Site
{
public:

    virtual ~Site() = default;

    virtual int dimension() const = 0;

    virtual bool is_fermionic() const { return false; }

    virtual LocalAction apply(
        const std::string& op,
        StateID local_state
    ) const = 0;
};

}