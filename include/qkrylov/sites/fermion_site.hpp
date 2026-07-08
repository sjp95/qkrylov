#pragma once

#include "site.hpp"

namespace qkrylov
{

class FermionSite : public Site
{
public:

    int dimension() const override { return 2; }
    bool is_fermionic() const override { return true; }

    LocalAction apply(
        const std::string& op,
        StateID local_state
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
