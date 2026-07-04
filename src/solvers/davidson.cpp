#include "qkrylov/solvers/davidson.hpp"

#include <random>
#include <algorithm>
#include <iostream>
#include <cmath>

namespace qkrylov
{

namespace
{
struct SmallEigensystem {
    std::vector<double> eigenvalues;
    std::vector<std::vector<Complex>> eigenvectors;
};

// Jacobi rotation-based eigensolver for small Hermitian matrices
SmallEigensystem solve_small_hermitian(const std::vector<std::vector<Complex>>& H) {
    int n = H.size();
    std::vector<std::vector<Complex>> A = H;
    std::vector<std::vector<Complex>> V(n, std::vector<Complex>(n, 0.0));
    for (int i = 0; i < n; ++i) V[i][i] = 1.0;

    for (int iter = 0; iter < 1000; ++iter) {
        double max_off = 0.0;
        int p = 0, q = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (std::abs(A[i][j]) > max_off) {
                    max_off = std::abs(A[i][j]);
                    p = i; q = j;
                }
            }
        }

        if (max_off < 1e-15) break;

        Complex app = A[p][p];
        Complex aqq = A[q][q];
        Complex apq = A[p][q];

        double phi = 0.5 * std::atan2(2.0 * std::abs(apq), std::real(app - aqq));
        Complex c = std::cos(phi);
        Complex s = std::sin(phi) * apq / std::abs(apq);

        for (int i = 0; i < n; ++i) {
            Complex vip = V[i][p];
            Complex viq = V[i][q];
            V[i][p] = c * vip + s * viq;
            V[i][q] = -std::conj(s) * vip + c * viq;

            if (i != p && i != q) {
                Complex aip = A[i][p];
                Complex aiq = A[i][q];
                A[i][p] = A[p][i] = c * aip + s * aiq;
                A[i][q] = A[q][i] = -std::conj(s) * aip + c * aiq;
            }
        }
        A[p][p] = c * c * app + s * std::conj(s) * aqq + 2.0 * std::real(c * s * std::conj(apq));
        A[q][q] = s * std::conj(s) * app + c * c * aqq - 2.0 * std::real(c * s * std::conj(apq));
        A[p][q] = A[q][p] = 0.0;
    }

    SmallEigensystem res;
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int i, int j) {
        return std::real(A[i][i]) < std::real(A[j][j]);
    });

    for (int i : idx) {
        res.eigenvalues.push_back(std::real(A[i][i]));
        std::vector<Complex> v(n);
        for (int j = 0; j < n; ++j) v[j] = V[j][i];
        res.eigenvectors.push_back(v);
    }
    return res;
}
}

DavidsonResult davidson_lowest(
    const MatrixFreeHamiltonian& H,
    int n_eig,
    int max_subspace,
    double tol
)
{
    const Index dim = H.dimension();
    if (dim == 0) return {};

    n_eig = std::min<int>(n_eig, dim);
    max_subspace = std::max(max_subspace, 2 * n_eig);

    Vector diag = H.diagonal();

    std::vector<Vector> V;
    std::vector<Vector> HV;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Initial subspace: use some diagonal dominance if possible,
    // or just unit vectors for the smallest diagonal elements
    std::vector<Index> diag_idx(dim);
    for(Index i=0; i<dim; ++i) diag_idx[i] = i;
    std::sort(diag_idx.begin(), diag_idx.end(), [&](Index i, Index j){
        return diag[i].real() < diag[j].real();
    });

    for (int i = 0; i < n_eig; ++i) {
        Vector v(dim, 0.0);
        v[diag_idx[i]] = 1.0;
        // Orthogonalize against previous V
        for (const auto& v_prev : V) axpy(-dot(v_prev, v), v_prev, v);
        double nrm = norm(v);
        if (nrm < 1e-10) {
            // Fallback to random if unit vector is not linearly independent
            for (Index j = 0; j < dim; ++j) v[j] = Complex(dist(rng), dist(rng));
            for (const auto& v_prev : V) axpy(-dot(v_prev, v), v_prev, v);
            nrm = norm(v);
        }
        scal(1.0/nrm, v);
        V.push_back(v);
        Vector hv(dim);
        H.apply(v, hv);
        HV.push_back(hv);
    }

    for (int iter = 0; iter < 100; ++iter) {
        int m = V.size();
        std::vector<std::vector<Complex>> H_sub(m, std::vector<Complex>(m));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j <= i; ++j) {
                H_sub[i][j] = dot(V[i], HV[j]);
                if (i != j) H_sub[j][i] = std::conj(H_sub[i][j]);
            }
        }

        auto eig = solve_small_hermitian(H_sub);

        bool all_converged = true;
        std::vector<Vector> next_corrections;

        for (int k = 0; k < n_eig; ++k) {
            double lambda = eig.eigenvalues[k];
            const auto& s = eig.eigenvectors[k];

            Vector r(dim, 0.0);
            for (int i = 0; i < m; ++i) {
                axpy(s[i], HV[i], r);
                axpy(-lambda * s[i], V[i], r);
            }

            double res_norm = norm(r);
            if (res_norm > tol) {
                all_converged = false;
                Vector t(dim);
                for (Index i = 0; i < dim; ++i) {
                    Complex diff = diag[i] - lambda;
                    if (std::abs(diff) < 1e-10) diff = (diff.real() >= 0) ? 1e-10 : -1e-10;
                    t[i] = r[i] / diff;
                }
                next_corrections.push_back(t);
            }
        }

        if (all_converged || iter == 99) {
            DavidsonResult res;
            for (int k = 0; k < n_eig; ++k) {
                res.eigenvalues.push_back(eig.eigenvalues[k]);
                Vector ev(dim, 0.0);
                for (int i = 0; i < m; ++i) axpy(eig.eigenvectors[k][i], V[i], ev);
                res.eigenvectors.push_back(ev);
            }
            return res;
        }

        // Expansion and restart
        if (V.size() + next_corrections.size() > (size_t)max_subspace) {
            std::vector<Vector> next_V;
            std::vector<Vector> next_HV;
            for (int k = 0; k < n_eig; ++k) {
                Vector ritz_v(dim, 0.0);
                Vector ritz_hv(dim, 0.0);
                for (int i = 0; i < m; ++i) {
                    axpy(eig.eigenvectors[k][i], V[i], ritz_v);
                    axpy(eig.eigenvectors[k][i], HV[i], ritz_hv);
                }
                next_V.push_back(ritz_v);
                next_HV.push_back(ritz_hv);
            }
            V = next_V;
            HV = next_HV;
        }

        for (auto& t : next_corrections) {
            for (const auto& v : V) axpy(-dot(v, t), v, t);
            double nrm = norm(t);
            if (nrm > 1e-10) {
                scal(1.0/nrm, t);
                V.push_back(t);
                Vector ht(dim);
                H.apply(t, ht);
                HV.push_back(ht);
            }
        }
    }

    return {};
}

}
