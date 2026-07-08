#pragma once

#include "site.hpp"

namespace qkrylov
{

class TJSite : public Site
{
public:

    int dimension() const override { return 3; }
    bool is_fermionic() const override { return true; }

    LocalAction apply(
        const std::string& op,
        StateID local_state
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
