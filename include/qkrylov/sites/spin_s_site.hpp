#pragma once

#include "qkrylov/core/types.hpp"
#include "qkrylov/sites/site.hpp"

#include <string>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {

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

} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov
