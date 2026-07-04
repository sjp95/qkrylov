#pragma once

#include "site.hpp"

namespace qkrylov
{

class FermionSite : public Site
{
public:

    LocalAction apply(
        const std::string& op,
        int site,
        StateID state
    ) const override;

private:

    static bool occupied(
        StateID state,
        int site
    );

    static double phase(
        StateID state,
        int site
    );
};

}
