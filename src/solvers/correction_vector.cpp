#include "qkrylov/solvers/correction_vector.hpp"

#include <cmath>
#include <iostream>

namespace qkrylov
{

CorrectionVectorResult correction_vector_spectral(
    const MatrixFreeHamiltonian& H,
    const Vector& Op_psi0,
    double E0,
    double omega,
    double eta,
    int max_iter,
    double tol
)
{
    const Index dim = H.dimension();
    if (dim == 0) return {{}, 0.0, 0, false};

    // Right hand side: b = eta * Op_psi0
    Vector b(dim);
    #pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(dim); ++i) {
        b[i] = eta * Op_psi0[i];
    }

    auto apply_A = [&](const Vector& y, Vector& Ay) {
        Vector z(dim, 0.0);
        H.apply(y.data(), z.data());
        const double shift = E0 + omega;
        #pragma omp parallel for
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(dim); ++i) {
            z[i] -= shift * y[i];
        }

        H.apply(z.data(), Ay.data());
        const double eta_sq = eta * eta;
        #pragma omp parallel for
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(dim); ++i) {
            Ay[i] = (Ay[i] - shift * z[i]) + eta_sq * y[i];
        }
    };

    // Conjugate Gradient (CG) solver for A x = b
    Vector x(dim, 0.0); // Initial guess
    Vector r = b;       // Initial residual r = b - A x0 = b
    Vector p = r;

    double rs_old = std::real(dot(r, r));
    const double b_norm = norm(b);
    if (b_norm < 1e-15) {
        return {x, 0.0, 0, true};
    }

    Vector Ap(dim);
    bool converged = false;
    int iter = 0;

    for (iter = 0; iter < max_iter; ++iter) {
        apply_A(p, Ap);

        Complex p_Ap = dot(p, Ap);
        double denom = std::real(p_Ap);
        if (std::abs(denom) < 1e-20) break;

        double alpha = rs_old / denom;

        axpy(alpha, p, x);
        axpy(-alpha, Ap, r);

        double rs_new = std::real(dot(r, r));
        if (std::sqrt(rs_new) / b_norm < tol) {
            converged = true;
            iter++;
            break;
        }

        double beta_cg = rs_new / rs_old;
        #pragma omp parallel for
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(dim); ++i) {
            p[i] = r[i] + beta_cg * p[i];
        }

        rs_old = rs_new;
    }

    // Spectral function S(omega) = (1/pi) * <Op_psi0 | Y>
    Complex inner = dot(Op_psi0, x);
    double spectral_func = (1.0 / M_PI) * std::real(inner);

    return {x, spectral_func, iter, converged};
}

}
