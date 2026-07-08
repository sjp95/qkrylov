#pragma once

#include "../core/types.hpp"

namespace qkrylov {

class Basis
{
public:

    virtual ~Basis() = default;

    virtual Index size() const = 0;

    virtual StateID state(Index i) const = 0;

    virtual Index index(StateID s) const = 0;

    virtual bool contains(StateID s) const = 0;

    virtual int nsites() const = 0;
};

}