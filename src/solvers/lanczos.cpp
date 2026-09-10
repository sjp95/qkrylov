#include "qkrylov/core/types.hpp"
#include "qkrylov/solvers/lanczos.hpp"

#include <random>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <iostream>
#include <cmath>
#include <Kokkos_Core.hpp>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {


namespace
{

struct TridiagResult {
    Real energy;
    std::vector<Real> eigenvector;
};

// Diagonalize symmetric tridiagonal matrix and return ground state energy and eigenvector
TridiagResult tridiag_ground_state_full(const std::vector<Real>& alpha, const std::vector<Real>& beta, int n)
{
    if (n == 0) return {0.0, {}};
    if (n == 1) return {alpha[0], {1.0}};

    std::vector<Real> d = alpha;
    std::vector<Real> e = beta;
    std::vector<std::vector<Real>> z(n, std::vector<Real>(n, 0.0));
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

        Real b = (d[m-1] - d[m]) / 2.0;
        Real c = e[m-1] * e[m-1];
        Real s = std::sqrt(b*b + c);
        Real shift = (b > 0) ? d[m] - c / (b + s) : d[m] - c / (b - s);

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

    int min_idx = 0;
    for (int i = 1; i < n; ++i) {
        if (d[i] < d[min_idx]) min_idx = i;
    }

    std::vector<Real> res_v(n);
    for (int i = 0; i < n; ++i) res_v[i] = z[i][min_idx];

    return {d[min_idx], res_v};
}
}

template <typename ExecSpace>
LanczosResult lanczos_ground_state(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    int maxiter,
    Real tol,
    bool two_pass
)
{
    const Index dim = H.dimension();
    if (dim == 0) return {};

    if (!two_pass) {
        VectorView<ExecSpace> v_prev("v_prev", dim);
        VectorView<ExecSpace> v_curr("v_curr", dim);
        VectorView<ExecSpace> w("w", dim);

        std::mt19937 rng(1234);
        std::uniform_real_distribution<Real> dist(-1.0, 1.0);

        auto v_curr_host = Kokkos::create_mirror_view(v_curr);
        for(Index i=0; i<dim; ++i) v_curr_host(i) = KComplex(dist(rng), dist(rng));
        Kokkos::deep_copy(v_curr, v_curr_host);
        normalize(v_curr);

        std::vector<VectorView<ExecSpace>> basis_vectors;
        VectorView<ExecSpace> v_curr_copy("basis_curr", dim);
        Kokkos::deep_copy(v_curr_copy, v_curr);
        basis_vectors.push_back(v_curr_copy);

        std::vector<Real> alphas;
        std::vector<Real> betas;

        bool is_converged = false;
        Real energy_old = 1e100;
        int actual_iters = 0;

        for(int iter=0; iter < std::min<int>(maxiter, dim); ++iter)
        {
            actual_iters = iter + 1;
            H.apply(v_curr, w);

            Real alpha = dot(v_curr, w).real();
            alphas.push_back(alpha);

            axpy(-alpha, v_curr, w);
            if (iter > 0) {
                axpy(-betas.back(), v_prev, w);
            }

            // Full reorthogonalization to maintain stability
            for (const auto& bv : basis_vectors) {
                axpy(-dot(bv, w), bv, w);
            }

            Real beta = norm(w);

            if (beta < 1e-15) {
                 is_converged = true;
                 break;
            }
            if (iter + 1 == std::min<int>(maxiter, dim)) {
                 break;
            }

            betas.push_back(beta);

            Kokkos::deep_copy(v_prev, v_curr);
            Kokkos::deep_copy(v_curr, w);
            scal(1.0/beta, v_curr);

            VectorView<ExecSpace> v_new("basis", dim);
            Kokkos::deep_copy(v_new, v_curr);
            basis_vectors.push_back(v_new);

            if (iter > 0) {
                auto tridiag = tridiag_ground_state_full(alphas, betas, alphas.size());
                if (std::abs(tridiag.energy - energy_old) < tol) {
                    energy_old = tridiag.energy;
                    is_converged = true;
                    break;
                }
                energy_old = tridiag.energy;
            }
        }

        auto final_tridiag = tridiag_ground_state_full(alphas, betas, alphas.size());

        LanczosResult res;
        res.energy = final_tridiag.energy;
        res.iterations = actual_iters;
        res.converged = is_converged;

        // Compute Ritz vector
        VectorView<ExecSpace> ritz("ritz", dim);
        for (int i = 0; i < (int)alphas.size(); ++i) {
            axpy(KComplex(final_tridiag.eigenvector[i], 0.0), basis_vectors[i], ritz);
        }
        normalize(ritz);

        copy_device_to_host(ritz, res.eigenvector);

        return res;
    }

    // Two-pass implementation
    // Pass 1: Compute Lanczos coefficients using 3 vectors (v_prev, v_curr, w)
    VectorView<ExecSpace> v_prev("v_prev", dim);
    VectorView<ExecSpace> v_curr("v_curr", dim);
    VectorView<ExecSpace> w("w", dim);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<Real> dist(-1.0, 1.0);

    auto v_curr_host = Kokkos::create_mirror_view(v_curr);
    for(Index i=0; i<dim; ++i) v_curr_host(i) = KComplex(dist(rng), dist(rng));
    Kokkos::deep_copy(v_curr, v_curr_host);
    normalize(v_curr);

    std::vector<Real> alphas;
    std::vector<Real> betas;

    bool is_converged = false;
    Real energy_old = 1e100;
    int actual_iters = 0;

    for(int iter=0; iter < std::min<int>(maxiter, dim); ++iter)
    {
        actual_iters = iter + 1;
        H.apply(v_curr, w);

        Real alpha = dot(v_curr, w).real();
        alphas.push_back(alpha);

        axpy(-alpha, v_curr, w);
        if (iter > 0) {
            axpy(-betas.back(), v_prev, w);
        }

        Real beta = norm(w);

        if (beta < 1e-15) {
             is_converged = true;
             break;
        }

        betas.push_back(beta);

        Kokkos::deep_copy(v_prev, v_curr);
        Kokkos::deep_copy(v_curr, w);
        scal(1.0/beta, v_curr);

        auto tridiag = tridiag_ground_state_full(alphas, betas, alphas.size());
        if (iter > 0 && std::abs(tridiag.energy - energy_old) < tol) {
            energy_old = tridiag.energy;
            is_converged = true;
            break;
        }
        if (iter + 1 == std::min<int>(maxiter, dim)) {
            is_converged = true;
            break;
        }
        energy_old = tridiag.energy;
    }

    auto final_tridiag = tridiag_ground_state_full(alphas, betas, alphas.size());

    LanczosResult res;
    res.energy = final_tridiag.energy;
    res.iterations = actual_iters;
    res.converged = is_converged;

    const int m = static_cast<int>(alphas.size());
    if (m == 0) return res;

    // Pass 2: Re-seed RNG, reconstruct Lanczos vectors on-the-fly and accumulate Ritz vector
    VectorView<ExecSpace> ritz("ritz", dim);
    zero_fill(ritz);

    rng.seed(1234);
    for(Index i=0; i<dim; ++i) v_curr_host(i) = KComplex(dist(rng), dist(rng));
    Kokkos::deep_copy(v_curr, v_curr_host);
    normalize(v_curr);
    zero_fill(v_prev);

    for (int k = 0; k < m; ++k) {
        // Accumulate z_k * v_k
        axpy(KComplex(final_tridiag.eigenvector[k], 0.0), v_curr, ritz);

        if (k + 1 < m) {
            H.apply(v_curr, w);
            axpy(-alphas[k], v_curr, w);
            if (k > 0) {
                axpy(-betas[k-1], v_prev, w);
            }
            Kokkos::deep_copy(v_prev, v_curr);
            Kokkos::deep_copy(v_curr, w);
            scal(1.0 / betas[k], v_curr);
        }
    }

    normalize(ritz);
    copy_device_to_host(ritz, res.eigenvector);

    return res;
}


// Explicit instantiations
#ifdef KOKKOS_ENABLE_SERIAL
template LanczosResult lanczos_ground_state<Kokkos::Serial>(const MatrixFreeHamiltonian<Kokkos::Serial>&, int, Real, bool);
#endif
#ifdef KOKKOS_ENABLE_OPENMP
template LanczosResult lanczos_ground_state<Kokkos::OpenMP>(const MatrixFreeHamiltonian<Kokkos::OpenMP>&, int, Real, bool);
#endif
#ifdef KOKKOS_ENABLE_THREADS
template LanczosResult lanczos_ground_state<Kokkos::Threads>(const MatrixFreeHamiltonian<Kokkos::Threads>&, int, Real, bool);
#endif
#ifdef KOKKOS_ENABLE_CUDA
template LanczosResult lanczos_ground_state<Kokkos::Cuda>(const MatrixFreeHamiltonian<Kokkos::Cuda>&, int, Real, bool);
#endif
#ifdef KOKKOS_ENABLE_HIP
template LanczosResult lanczos_ground_state<Kokkos::HIP>(const MatrixFreeHamiltonian<Kokkos::HIP>&, int, Real, bool);
#endif
#ifdef KOKKOS_ENABLE_SYCL
template LanczosResult lanczos_ground_state<Kokkos::Experimental::SYCL>(const MatrixFreeHamiltonian<Kokkos::Experimental::SYCL>&, int, Real, bool);
#endif
}

}
