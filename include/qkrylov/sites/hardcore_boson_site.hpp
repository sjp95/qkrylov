#pragma once

#include "site.hpp"

namespace qkrylov
{

class HardcoreBosonSite : public Site
{
public:

    int dimension() const override { return 2; }

    LocalAction apply(
        const std::string& op,
        StateID local_state
    ) const override;
};

}
