#pragma once

#include "site.hpp"

namespace qkrylov
{

class BosonSite : public Site
{
public:

    BosonSite(int max_occupancy);

    int dimension() const override { return max_occ_ + 1; }

    LocalAction apply(
        const std::string& op,
        StateID local_state
    ) const override;

private:

    int max_occ_;
};

}
