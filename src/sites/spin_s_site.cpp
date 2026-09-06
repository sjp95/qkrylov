#include "qkrylov/sites/spin_s_site.hpp"

#include <stdexcept>
#include <cmath>

namespace qkrylov
{

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

    const int sigma = static_cast<int>((state / d_pow) % d_);
    const double mz = static_cast<double>(sigma) - S_;

    if (op == "Sz") {
        a.valid = true;
        a.new_state = state;
        a.matrix_element = mz;
        return a;
    }

    if (op == "Sp") {
        if (sigma >= d_ - 1) return a;
        a.valid = true;
        a.new_state = state + d_pow;
        a.matrix_element = std::sqrt(S_ * (S_ + 1.0) - mz * (mz + 1.0));
        return a;
    }

    if (op == "Sm") {
        if (sigma <= 0) return a;
        a.valid = true;
        a.new_state = state - d_pow;
        a.matrix_element = std::sqrt(S_ * (S_ + 1.0) - mz * (mz - 1.0));
        return a;
    }

    if (op == "Sx") {
        // Sx = 0.5 * (Sp + Sm)
        // Expressed as action by returning Sp/Sm dynamically or handling Sx directly if state allows:
        // Here we handle Sx as linear combo components if single action, but standard OpSum expands or can handle Sx via Sp + Sm.
        // For local action interface returning a single state:
        // Note: Sx flips state up or down with appropriate weight if state is pure basis. But Sx acts on state producing 2 basis states!
        // In ED, Sx should be expanded into 0.5 Sp + 0.5 Sm in OpSum.
        throw std::runtime_error("SpinSSite: For S > 1/2, please express Sx as 0.5*Sp + 0.5*Sm in OpSum.");
    }

    if (op == "Sy") {
        throw std::runtime_error("SpinSSite: For S > 1/2, please express Sy as -0.5i*Sp + 0.5i*Sm in OpSum.");
    }

    throw std::runtime_error("Unknown spin operator for SpinSSite: " + op);
}

}
