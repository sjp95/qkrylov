#pragma once

#include "site.hpp"

namespace qkrylov
{

class SpinSSite : public Site
{
public:

    SpinSSite(double S);

    int dimension() const override { return dim_; }

    LocalAction apply(
        const std::string& op,
        StateID local_state
    ) const override;

private:

    double S_;
    int dim_;
};

}
