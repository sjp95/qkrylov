#pragma once

#include "site.hpp"

namespace qkrylov
{

class SpinHalfSite : public Site
{
public:

    LocalAction apply(
        const std::string& op,
        int site,
        StateID state
    ) const override;

private:

    static bool spin_up(
        StateID state,
        int site
    );
};

}