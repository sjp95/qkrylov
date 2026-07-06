#include "qkrylov/solvers/lanczos.hpp"

#include <random>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <iostream>
#include <cmath>

namespace qkrylov
{

namespace
{

struct TridiagResult {
    double energy;
    std::vector<double> eigenvector;
};

// Diagonalize symmetric tridiagonal matrix and return ground state energy and eigenvector
TridiagResult tridiag_ground_state_full(const std::vector<double>& alpha, const std::vector<double>& beta, int n)
{
    if (n == 0) return {0.0, {}};
    if (n == 1) return {alpha[0], {1.0}};

    std::vector<double> d = alpha;
    std::vector<double> e = beta;
    std::vector<std::vector<double>> z(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) z[i][i] = 1.0;

    for (int iter = 0; iter < 1000; ++iter) {
        for (int i = 0; i < n - 1; ++i) {
            if (std::abs(e[i]) < 1e-14 * (std::abs(d[i]) + std::abs(d[i+1]))) {
                e[i] = 0.0;
            }
        }

        int m = n - 1;
        while (m > 0 && e[m-1] == 0.0) m--;
        if (m == 0) break;

        int l = m - 1;
        while (l > 0 && e[l-1] != 0.0) l--;

        double b = (d[m-1] - d[m]) / 2.0;
        double c = e[m-1] * e[m-1];
        double s = std::sqrt(b*b + c);
        double shift = (b > 0) ? d[m] - c / (b + s) : d[m] - c / (b - s);

        double p = d[l] - shift;
        double g = e[l];

        for (int i = l; i < m; ++i) {
            double r = std::hypot(p, g);
            double cos_theta = p / r;
            double sin_theta = g / r;

            if (i > l) e[i-1] = r;

            double f = cos_theta * d[i] + sin_theta * e[i];
            double g_next = cos_theta * e[i] + sin_theta * d[i+1];
            double h = sin_theta * d[i] - cos_theta * e[i];
            double k = sin_theta * e[i] - cos_theta * d[i+1];

            d[i] = cos_theta * f + sin_theta * g_next;
            e[i] = cos_theta * h + sin_theta * k;
            d[i+1] = sin_theta * h - cos_theta * k;

            // Update eigenvectors z
            for (int j = 0; j < n; ++j) {
                double z1 = z[j][i];
                double z2 = z[j][i+1];
                z[j][i] = cos_theta * z1 + sin_theta * z2;
                z[j][i+1] = sin_theta * z1 - cos_theta * z2;
            }

            if (i < m - 1) {
                p = e[i];
                g = sin_theta * e[i+1];
                e[i+1] = -cos_theta * e[i+1];
            }
        }
    }

    int min_idx = 0;
    for (int i = 1; i < n; ++i) {
        if (d[i] < d[min_idx]) min_idx = i;
    }

    std::vector<double> res_v(n);
    for (int i = 0; i < n; ++i) res_v[i] = z[i][min_idx];

    return {d[min_idx], res_v};
}
}

LanczosResult lanczos_ground_state(
    const MatrixFreeHamiltonian& H,
    int maxiter,
    double tol
)
{
    const Index dim = H.dimension();
    if (dim == 0) return {};

    Vector v_prev(dim, 0.0);
    Vector v_curr(dim, 0.0);
    Vector w(dim, 0.0);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for(Index i=0; i<dim; ++i) v_curr[i] = Complex(dist(rng), dist(rng));
    normalize(v_curr);

    std::vector<Vector> basis_vectors;
    basis_vectors.push_back(v_curr);

    std::vector<double> alphas;
    std::vector<double> betas;

    double energy_old = 1e100;
    int actual_iters = 0;

    for(int iter=0; iter < std::min<int>(maxiter, dim); ++iter)
    {
        actual_iters = iter + 1;
        H.apply(v_curr, w);

        double alpha = std::real(dot(v_curr, w));
        alphas.push_back(alpha);

        axpy(-alpha, v_curr, w);
        if (iter > 0) {
            axpy(-betas.back(), v_prev, w);
        }

        // Full reorthogonalization to maintain stability
        for (const auto& bv : basis_vectors) {
            axpy(-dot(bv, w), bv, w);
        }

        double beta = norm(w);

        if (beta < 1e-15 || iter + 1 == std::min<int>(maxiter, dim)) {
             break;
        }

        betas.push_back(beta);

        v_prev = v_curr;
        v_curr = w;
        scal(1.0/beta, v_curr);
        basis_vectors.push_back(v_curr);

        if (iter > 0) {
            // Check convergence only every few iterations or after some initial steps
            auto tridiag = tridiag_ground_state_full(alphas, betas, alphas.size());
            if (std::abs(tridiag.energy - energy_old) < tol) {
                energy_old = tridiag.energy;
                break;
            }
            energy_old = tridiag.energy;
        }
    }

    auto final_tridiag = tridiag_ground_state_full(alphas, betas, alphas.size());

    LanczosResult res;
    res.energy = final_tridiag.energy;

    // Compute Ritz vector
    res.eigenvector.assign(dim, 0.0);
    for (int i = 0; i < (int)alphas.size(); ++i) {
        axpy(final_tridiag.eigenvector[i], basis_vectors[i], res.eigenvector);
    }
    normalize(res.eigenvector);

    return res;
}

}
