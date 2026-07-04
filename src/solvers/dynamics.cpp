#include "qkrylov/solvers/dynamics.hpp"

#include <cmath>
#include <complex>

namespace qkrylov
{

DynamicsResult continued_fraction_coeffs(
    const MatrixFreeHamiltonian& H,
    const Vector& phi0,
    int n_iter
)
{
    const Index dim = H.dimension();
    if (dim == 0) return {};

    double norm_phi = norm(phi0);
    if (norm_phi < 1e-15) return { {}, {}, 0.0 };

    Vector v_curr = phi0;
    scal(1.0/norm_phi, v_curr);

    Vector v_prev(dim, 0.0);
    Vector w(dim, 0.0);

    DynamicsResult res;
    res.norm_phi0 = norm_phi;

    for (int iter = 0; iter < n_iter; ++iter) {
        H.apply(v_curr, w);

        double alpha = std::real(dot(v_curr, w));
        res.alphas.push_back(alpha);

        axpy(-alpha, v_curr, w);
        if (iter > 0) {
            axpy(-res.betas.back(), v_prev, w);
        }

        double beta = norm(w);
        if (beta < 1e-15) break;

        res.betas.push_back(beta);
        v_prev = v_curr;
        v_curr = w;
        scal(1.0/beta, v_curr);
    }

    return res;
}

double evaluate_spectral_function(
    const DynamicsResult& res,
    double omega,
    double E0,
    double eta
)
{
    if (res.alphas.empty()) return 0.0;

    const int n = res.alphas.size();
    std::complex<double> z(omega + E0, eta);

    // Backward recursion for continued fraction
    // C_n = 1 / (z - a_n)
    // C_{i} = 1 / (z - a_i - b_i^2 * C_{i+1})

    std::complex<double> f = 0.0;
    for (int i = n - 1; i >= 0; --i) {
        if (i == n - 1) {
            f = 1.0 / (z - res.alphas[i]);
        } else {
            f = 1.0 / (z - res.alphas[i] - res.betas[i] * res.betas[i] * f);
        }
    }

    return -1.0 / M_PI * std::imag(res.norm_phi0 * res.norm_phi0 * f);
}

}
