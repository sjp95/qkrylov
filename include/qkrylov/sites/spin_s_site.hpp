#pragma once

#include "site.hpp"

namespace qkrylov
{

class SpinSSite : public Site
{
public:

    explicit SpinSSite(double S = 0.5);

    LocalAction apply(
        const std::string& op,
        int site,
        StateID state
    ) const override;

    double spin() const noexcept { return S_; }
    int dimension_per_site() const noexcept { return d_; }

private:

    double S_;
    int d_;
};

}
