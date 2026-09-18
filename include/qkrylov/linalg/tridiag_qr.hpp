#pragma once

#include "qkrylov/core/types.hpp"

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {
namespace linalg {

struct TridiagAllEigensystem {
    std::vector<Real> eigenvalues;
    std::vector<std::vector<Real>> eigenvectors;
};

struct TridiagGroundState {
    Real energy = Real(0.0);
    std::vector<Real> eigenvector;
};

/// Diagonalize a symmetric tridiagonal matrix defined by diagonal alpha and off-diagonal beta.
/// Uses implicit QR iterations with Wilkinson shifts and Givens plane rotations.
/// Returns all eigenvalues and eigenvectors sorted in ascending order.
inline TridiagAllEigensystem tridiag_eigensystem_full(
    const std::vector<Real>& alpha,
    const std::vector<Real>& beta,
    int n
) {
    if (n <= 0) return {{}, {}};
    if (n == 1) return {{alpha[0]}, {{Real(1.0)}}};

    std::vector<Real> d(alpha.begin(), alpha.begin() + n);
    std::vector<Real> e(n, Real(0.0));
    for (int i = 0; i < n - 1 && i < static_cast<int>(beta.size()); ++i) {
        e[i] = beta[i];
    }

    std::vector<std::vector<Real>> z(n, std::vector<Real>(n, Real(0.0)));
    for (int i = 0; i < n; ++i) z[i][i] = Real(1.0);

    const Real eps = std::numeric_limits<Real>::epsilon() * Real(4.0);
    const int max_qr_iter = std::max(1000, 100 * n);

    for (int iter = 0; iter < max_qr_iter; ++iter) {
        for (int i = 0; i < n - 1; ++i) {
            if (std::abs(e[i]) <= eps * (std::abs(d[i]) + std::abs(d[i+1]))) {
                e[i] = Real(0.0);
            }
        }

        int m = n - 1;
        while (m > 0 && e[m-1] == Real(0.0)) m--;
        if (m == 0) break;

        int l = m - 1;
        while (l > 0 && e[l-1] != Real(0.0)) l--;

        Real b = (d[m-1] - d[m]) / Real(2.0);
        Real c = e[m-1] * e[m-1];
        Real s = std::sqrt(b*b + c);
        Real shift = (b > Real(0.0)) ? d[m] - c / (b + s) : d[m] - c / (b - s);

        Real p = d[l] - shift;
        Real g = e[l];

        for (int i = l; i < m; ++i) {
            Real r = std::hypot(p, g);
            Real cos_theta = p / r;
            Real sin_theta = g / r;

            if (i > l) e[i-1] = r;

            Real f = cos_theta * d[i] + sin_theta * e[i];
            Real g_next = cos_theta * e[i] + sin_theta * d[i+1];
            Real h = sin_theta * d[i] - cos_theta * e[i];
            Real k = sin_theta * e[i] - cos_theta * d[i+1];

            d[i] = cos_theta * f + sin_theta * g_next;
            e[i] = cos_theta * h + sin_theta * k;
            d[i+1] = sin_theta * h - cos_theta * k;

            // Update eigenvectors z
            for (int j = 0; j < n; ++j) {
                Real z1 = z[j][i];
                Real z2 = z[j][i+1];
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

    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return d[a] < d[b];
    });

    TridiagAllEigensystem res;
    res.eigenvalues.resize(n);
    res.eigenvectors.resize(n, std::vector<Real>(n));
    for (int k = 0; k < n; ++k) {
        int col = idx[k];
        res.eigenvalues[k] = d[col];
        for (int j = 0; j < n; ++j) {
            res.eigenvectors[k][j] = z[j][col];
        }
    }

    return res;
}

/// Convenience function returning only ground state eigenvalue and eigenvector.
inline TridiagGroundState tridiag_ground_state_full(
    const std::vector<Real>& alpha,
    const std::vector<Real>& beta,
    int n
) {
    auto all = tridiag_eigensystem_full(alpha, beta, n);
    if (all.eigenvalues.empty()) return {Real(0.0), {}};
    return {all.eigenvalues[0], all.eigenvectors[0]};
}

} // namespace linalg
} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov
