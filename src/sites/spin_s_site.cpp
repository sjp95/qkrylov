#include "qkrylov/core/types.hpp"
#include "qkrylov/sites/spin_s_site.hpp"

#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {

SpinSSite::SpinSSite(double S)
    : S_(S),
      d_(static_cast<int>(std::round(2.0 * S + 1.0)))
{
    if (S <= 0.0) {
        throw std::runtime_error("SpinSSite: Spin S must be positive (e.g., 0.5, 1.0, 1.5, 2.0, 2.5).");
    }
}

LocalAction SpinSSite::apply(
    const std::string& op,
    int site,
    StateID state
) const
{
    LocalAction a;

    // Base-d state representation:
    // Local state sigma_j in {0, 1, ..., d-1} where m_z = sigma_j - S
    // For site j, state has factor d^j
    StateID d_pow = 1;
    for (int i = 0; i < site; ++i) {
        d_pow *= static_cast<StateID>(d_);
    }

    const int sigma = static_cast<int>((state / d_pow) % static_cast<StateID>(d_));
    const double mz = static_cast<double>(sigma) - S_;

    if (op == "Sz") {
        a.valid = true;
        a.new_state = state;
        a.matrix_element = static_cast<Real>(mz);
        return a;
    }

    if (op == "Sp") {
        if (sigma >= d_ - 1) {
            return a;
        }
        a.valid = true;
        a.new_state = state + d_pow;
        a.matrix_element = static_cast<Real>(std::sqrt(std::max(0.0, S_ * (S_ + 1.0) - mz * (mz + 1.0))));
        return a;
    }

    if (op == "Sm") {
        if (sigma <= 0) {
            return a;
        }
        a.valid = true;
        a.new_state = state - d_pow;
        a.matrix_element = static_cast<Real>(std::sqrt(std::max(0.0, S_ * (S_ + 1.0) - mz * (mz - 1.0))));
        return a;
    }

    if (op == "Sx") {
        throw std::runtime_error("SpinSSite: For S > 1/2, please express Sx as 0.5*Sp + 0.5*Sm in OpSum.");
    }

    if (op == "Sy") {
        throw std::runtime_error("SpinSSite: For S > 1/2, please express Sy as -0.5i*Sp + 0.5i*Sm in OpSum.");
    }

    throw std::runtime_error("Unknown spin operator for SpinSSite: " + op);
}

} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov
