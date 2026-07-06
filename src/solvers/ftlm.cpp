#include "qkrylov/solvers/ftlm.hpp"
#include "qkrylov/linalg/vector_ops.hpp"

#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace qkrylov
{

namespace
{
// Helper for tridiagonalization (simple version without Ritz vector storage)
struct Tridiag
{
    std::vector<double> alphas;
    std::vector<double> betas;
};

Tridiag compute_tridiag(const MatrixFreeHamiltonian& H, const Vector& v_start, int n_steps)
{
    const Index dim = H.dimension();
    Vector v_curr = v_start;
    normalize(v_curr);

    Vector v_prev(dim, 0.0);
    Vector w(dim, 0.0);
    Tridiag res;

    for (int i = 0; i < n_steps; ++i) {
        H.apply(v_curr, w);
        double alpha = std::real(dot(v_curr, w));
        res.alphas.push_back(alpha);

        axpy(-alpha, v_curr, w);
        if (i > 0) axpy(-res.betas.back(), v_prev, w);

        double beta = norm(w);
        if (beta < 1e-15) break;
        res.betas.push_back(beta);

        v_prev = v_curr;
        v_curr = w;
        scal(1.0/beta, v_curr);
    }
    return res;
}

// Simple tridiagonal diagonalization to get all eigenvalues and first components of eigenvectors
struct FullTridiagResult {
    std::vector<double> eigenvalues;
    std::vector<double> first_components;
};

FullTridiagResult diagonalize_tridiag_components(const std::vector<double>& alpha, const std::vector<double>& beta)
{
    int n = alpha.size();
    if (n == 0) return {};
    if (n == 1) return { {alpha[0]}, {1.0} };

    std::vector<double> d = alpha;
    std::vector<double> e = beta;
    std::vector<std::vector<double>> z(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) z[i][i] = 1.0;

    for (int iter = 0; iter < 1000; ++iter) {
        for (int i = 0; i < n - 1; ++i) {
            if (std::abs(e[i]) < 1e-14 * (std::abs(d[i]) + std::abs(d[i+1]))) e[i] = 0.0;
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

    FullTridiagResult res;
    for (int i = 0; i < n; ++i) {
        res.eigenvalues.push_back(d[i]);
        res.first_components.push_back(z[0][i]);
    }
    return res;
}
}

FTLMResult ftlm(
    const MatrixFreeHamiltonian& H,
    double beta,
    int n_random,
    int n_steps
)
{
    const Index dim = H.dimension();
    if (dim == 0) return {beta};

    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    double Z = 0.0;
    double E = 0.0;
    double E2 = 0.0;

    for (int r = 0; r < n_random; ++r) {
        Vector r_vec(dim);
        for (Index i = 0; i < dim; ++i) r_vec[i] = Complex(dist(rng), dist(rng));
        double nrm = norm(r_vec);

        auto tridiag = compute_tridiag(H, r_vec, n_steps);
        auto eig = diagonalize_tridiag_components(tridiag.alphas, tridiag.betas);

        for (size_t i = 0; i < eig.eigenvalues.size(); ++i) {
            double weight = nrm * nrm * eig.first_components[i] * eig.first_components[i] * std::exp(-beta * eig.eigenvalues[i]);
            Z += weight;
            E += eig.eigenvalues[i] * weight;
            E2 += eig.eigenvalues[i] * eig.eigenvalues[i] * weight;
        }
    }

    Z /= n_random;
    E /= n_random;
    E2 /= n_random;

    FTLMResult res;
    res.beta = beta;
    res.partition_function = Z;
    res.internal_energy = E / Z;
    res.specific_heat = (beta * beta) * (E2 / Z - (E / Z) * (E / Z));

    return res;
}

}
