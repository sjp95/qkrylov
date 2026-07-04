#pragma once

#include "site.hpp"

namespace qkrylov
{

class TJSite : public Site
{
public:

    LocalAction apply(
        const std::string& op,
        int site,
        StateID state
    ) const override;

private:

    static bool occupied_up(
        StateID state,
        int site
    );

    static bool occupied_dn(
        StateID state,
        int site
    );

    static double phase_up(
        StateID state,
        int site
    );

    static double phase_dn(
        StateID state,
        int site
    );
};

}
