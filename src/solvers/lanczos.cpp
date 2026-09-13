#include "qkrylov/core/types.hpp"
#include "qkrylov/solvers/lanczos.hpp"

#include <random>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <iostream>
#include <cmath>
#include <limits>
#include <Kokkos_Core.hpp>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {


namespace
{

struct TridiagAllEigensystem {
    std::vector<Real> eigenvalues;
    std::vector<std::vector<Real>> eigenvectors;
};

// Diagonalize symmetric tridiagonal matrix and return all eigenvalues and eigenvectors sorted in ascending order
TridiagAllEigensystem tridiag_eigensystem_full(const std::vector<Real>& alpha, const std::vector<Real>& beta, int n)
{
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

struct TridiagResult {
    Real energy;
    std::vector<Real> eigenvector;
};

// Diagonalize symmetric tridiagonal matrix and return ground state energy and eigenvector
TridiagResult tridiag_ground_state_full(const std::vector<Real>& alpha, const std::vector<Real>& beta, int n)
{
    auto all = tridiag_eigensystem_full(alpha, beta, n);
    if (all.eigenvalues.empty()) return {Real(0.0), {}};
    return {all.eigenvalues[0], all.eigenvectors[0]};
}
} // namespace

} // namespace QKRYLOV_PRECISION_NAMESPACE

namespace solvers {

template <typename Policy, typename ExecSpace>
QKRYLOV_PRECISION_NAMESPACE::LanczosResult lanczos(
    const QKRYLOV_PRECISION_NAMESPACE::MatrixFreeHamiltonian<ExecSpace>& H,
    const LanczosConfig& config
)
{
    using namespace QKRYLOV_PRECISION_NAMESPACE;
    constexpr bool is_two_pass = std::is_same_v<Policy, policy::TwoPass>;
    static_assert(
        std::is_same_v<Policy, policy::Default> ||
        std::is_same_v<Policy, policy::SinglePass> ||
        std::is_same_v<Policy, policy::TwoPass>,
        "Unknown solver policy"
    );

    const Index dim = H.dimension();
    if (dim == 0) return {};

    int maxiter = config.maxiter;
    Real tol = config.tol;

    VectorView<ExecSpace> v_prev("v_prev", dim);
    VectorView<ExecSpace> v_curr("v_curr", dim);
    VectorView<ExecSpace> w("w", dim);

    const Real mach_eps = std::numeric_limits<Real>::epsilon() * Real(4.0);

    const uint32_t seed = 1234;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<Real> dist(-1.0, 1.0);
    
    auto v_curr_host = Kokkos::create_mirror_view(v_curr);
    if (!config.initial_vector.empty()) {
        if (static_cast<Index>(config.initial_vector.size()) != dim) {
            throw std::invalid_argument("initial_vector size (" + std::to_string(config.initial_vector.size()) +
                                       ") does not match Hamiltonian dimension (" + std::to_string(dim) + ")");
        }
        for (Index i = 0; i < dim; ++i) {
            v_curr_host(i) = KComplex(config.initial_vector[i].real(), config.initial_vector[i].imag());
        }
        Kokkos::deep_copy(v_curr, v_curr_host);
        Real init_norm = norm(v_curr);
        if (init_norm < mach_eps) {
            throw std::invalid_argument("initial_vector has zero norm");
        }
        normalize(v_curr);
    } else {
        for(Index i=0; i<dim; ++i) v_curr_host(i) = KComplex(dist(rng), dist(rng));
        Kokkos::deep_copy(v_curr, v_curr_host);
        normalize(v_curr);
    }

    std::vector<VectorView<ExecSpace>> basis_vectors;
    if constexpr (!is_two_pass) {
        VectorView<ExecSpace> v_curr_copy("basis_curr", dim);
        Kokkos::deep_copy(v_curr_copy, v_curr);
        basis_vectors.push_back(v_curr_copy);
    }

    std::vector<Real> alphas;
    std::vector<Real> betas;

    bool is_converged = false;
    Real energy_old = std::numeric_limits<Real>::infinity();
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

        if constexpr (!is_two_pass) {
            // DGKS full reorthogonalization ("twice is enough") to maintain stability to machine precision
            for (int pass = 0; pass < 2; ++pass) {
                for (const auto& bv : basis_vectors) {
                    axpy(-dot(bv, w), bv, w);
                }
            }
        }

        Real beta = norm(w);

        if (beta < mach_eps) {
             is_converged = true;
             break;
        }
        if (iter + 1 == dim) {
             is_converged = true;
             break;
        }
        if (iter + 1 == maxiter) {
             break;
        }

        betas.push_back(beta);

        Kokkos::deep_copy(v_prev, v_curr);
        Kokkos::deep_copy(v_curr, w);
        scal(1.0/beta, v_curr);
        
        if constexpr (!is_two_pass) {
            VectorView<ExecSpace> v_new("basis", dim);
            Kokkos::deep_copy(v_new, v_curr);
            basis_vectors.push_back(v_new);
        }

        if (iter > 0) {
            // Check convergence only every few iterations or after some initial steps
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

    if constexpr (!is_two_pass) {
        // Compute Ritz vector using single pass
        VectorView<ExecSpace> ritz("ritz", dim);
        for (int i = 0; i < (int)alphas.size(); ++i) {
            axpy(KComplex(final_tridiag.eigenvector[i], 0.0), basis_vectors[i], ritz);
        }
        normalize(ritz);
        copy_device_to_host(ritz, res.eigenvector);
        return res;
    } else {
        // Two-pass reconstruction
        VectorView<ExecSpace> ritz("ritz", dim);
        Kokkos::deep_copy(ritz, KComplex(0.0, 0.0));

        if (!config.initial_vector.empty()) {
            for (Index i = 0; i < dim; ++i) {
                v_curr_host(i) = KComplex(config.initial_vector[i].real(), config.initial_vector[i].imag());
            }
            Kokkos::deep_copy(v_curr, v_curr_host);
            normalize(v_curr);
        } else {
            rng.seed(seed);
            for(Index i=0; i<dim; ++i) v_curr_host(i) = KComplex(dist(rng), dist(rng));
            Kokkos::deep_copy(v_curr, v_curr_host);
            normalize(v_curr);
        }
        
        Kokkos::deep_copy(v_prev, KComplex(0.0, 0.0));

        int m = static_cast<int>(alphas.size());
        for (int iter = 0; iter < m; ++iter) {
            axpy(KComplex(final_tridiag.eigenvector[iter], 0.0), v_curr, ritz);

            if (iter + 1 == m) break;

            H.apply(v_curr, w);
            axpy(-alphas[iter], v_curr, w);
            if (iter > 0) {
                axpy(-betas[iter-1], v_prev, w);
            }

            Kokkos::deep_copy(v_prev, v_curr);
            Kokkos::deep_copy(v_curr, w);
            scal(1.0 / betas[iter], v_curr);
        }

        normalize(ritz);
        copy_device_to_host(ritz, res.eigenvector);

        return res;
    }
}

template <typename ExecSpace>
QKRYLOV_PRECISION_NAMESPACE::LanczosLowestResult lanczos_lowest(
    const QKRYLOV_PRECISION_NAMESPACE::MatrixFreeHamiltonian<ExecSpace>& H,
    const LanczosLowestConfig& config
)
{
    using namespace QKRYLOV_PRECISION_NAMESPACE;
    const Index dim = H.dimension();
    if (dim == 0) return {};

    int n_eig = config.n_eig;
    n_eig = std::min<int>(n_eig, static_cast<int>(dim));
    if (n_eig <= 0) return {};

    int maxiter = std::max(config.maxiter, n_eig);
    Real tol = config.tol;

    VectorView<ExecSpace> v_prev("v_prev", dim);
    VectorView<ExecSpace> v_curr("v_curr", dim);
    VectorView<ExecSpace> w("w", dim);

    const Real mach_eps = std::numeric_limits<Real>::epsilon() * Real(4.0);

    auto v_curr_host = Kokkos::create_mirror_view(v_curr);
    if (!config.initial_vector.empty()) {
        if (static_cast<Index>(config.initial_vector.size()) != dim) {
            throw std::invalid_argument("initial_vector size (" + std::to_string(config.initial_vector.size()) +
                                       ") does not match Hamiltonian dimension (" + std::to_string(dim) + ")");
        }
        for (Index i = 0; i < dim; ++i) {
            v_curr_host(i) = KComplex(config.initial_vector[i].real(), config.initial_vector[i].imag());
        }
        Kokkos::deep_copy(v_curr, v_curr_host);
        Real init_norm = norm(v_curr);
        if (init_norm < mach_eps) {
            throw std::invalid_argument("initial_vector has zero norm");
        }
        normalize(v_curr);
    } else {
        const uint32_t seed = 1234;
        std::mt19937 rng(seed);
        std::uniform_real_distribution<Real> dist(-1.0, 1.0);
        for (Index i = 0; i < dim; ++i) v_curr_host(i) = KComplex(dist(rng), dist(rng));
        Kokkos::deep_copy(v_curr, v_curr_host);
        normalize(v_curr);
    }

    std::vector<VectorView<ExecSpace>> basis_vectors;
    VectorView<ExecSpace> v_curr_copy("basis_curr", dim);
    Kokkos::deep_copy(v_curr_copy, v_curr);
    basis_vectors.push_back(v_curr_copy);

    std::vector<Real> alphas;
    std::vector<Real> betas;

    bool is_converged = false;
    std::vector<Real> prev_energies;
    int actual_iters = 0;

    for (int iter = 0; iter < std::min<int>(maxiter, static_cast<int>(dim)); ++iter)
    {
        actual_iters = iter + 1;
        H.apply(v_curr, w);

        Real alpha = dot(v_curr, w).real();
        alphas.push_back(alpha);

        axpy(-alpha, v_curr, w);
        if (iter > 0) {
            axpy(-betas.back(), v_prev, w);
        }

        // DGKS twice-is-enough full reorthogonalization
        for (int pass = 0; pass < 2; ++pass) {
            for (const auto& bv : basis_vectors) {
                axpy(-dot(bv, w), bv, w);
            }
        }

        Real beta = norm(w);

        if (beta < mach_eps) {
            is_converged = true;
            break;
        }
        if (iter + 1 == static_cast<int>(dim) || iter + 1 == maxiter) {
            break;
        }

        betas.push_back(beta);

        Kokkos::deep_copy(v_prev, v_curr);
        Kokkos::deep_copy(v_curr, w);
        scal(Real(1.0) / beta, v_curr);

        VectorView<ExecSpace> v_new("basis", dim);
        Kokkos::deep_copy(v_new, v_curr);
        basis_vectors.push_back(v_new);

        int m = static_cast<int>(alphas.size());
        if (m >= n_eig) {
            auto tridiag = tridiag_eigensystem_full(alphas, betas, m);
            if (!prev_energies.empty()) {
                bool all_converged = true;
                for (int k = 0; k < n_eig; ++k) {
                    Real diff = std::abs(tridiag.eigenvalues[k] - prev_energies[k]);
                    Real ritz_res = beta * std::abs(tridiag.eigenvectors[k][m - 1]);
                    if (diff > tol && ritz_res > tol) {
                        all_converged = false;
                        break;
                    }
                }
                if (all_converged) {
                    is_converged = true;
                    break;
                }
            }
            prev_energies.assign(tridiag.eigenvalues.begin(), tridiag.eigenvalues.begin() + n_eig);
        }
    }

    auto final_tridiag = tridiag_eigensystem_full(alphas, betas, static_cast<int>(alphas.size()));

    LanczosLowestResult res;
    res.iterations = actual_iters;
    res.converged = is_converged;

    int num_out = std::min<int>(n_eig, static_cast<int>(final_tridiag.eigenvalues.size()));
    res.eigenvalues.assign(final_tridiag.eigenvalues.begin(), final_tridiag.eigenvalues.begin() + num_out);

    if (config.compute_eigenvectors) {
        for (int k = 0; k < num_out; ++k) {
            VectorView<ExecSpace> ritz("ritz", dim);
            for (int i = 0; i < static_cast<int>(alphas.size()); ++i) {
                axpy(KComplex(final_tridiag.eigenvectors[k][i], 0.0), basis_vectors[i], ritz);
            }
            normalize(ritz);
            HostVector host_ritz;
            copy_device_to_host(ritz, host_ritz);
            res.eigenvectors.push_back(std::move(host_ritz));
        }
    }

    return res;
}

} // namespace solvers

// Explicit instantiations
#define INSTANTIATE_LANCZOS(Space) \
    template QKRYLOV_PRECISION_NAMESPACE::LanczosResult solvers::lanczos<solvers::policy::Default, Space>(const QKRYLOV_PRECISION_NAMESPACE::MatrixFreeHamiltonian<Space>&, const QKRYLOV_PRECISION_NAMESPACE::LanczosConfig&); \
    template QKRYLOV_PRECISION_NAMESPACE::LanczosResult solvers::lanczos<solvers::policy::SinglePass, Space>(const QKRYLOV_PRECISION_NAMESPACE::MatrixFreeHamiltonian<Space>&, const QKRYLOV_PRECISION_NAMESPACE::LanczosConfig&); \
    template QKRYLOV_PRECISION_NAMESPACE::LanczosResult solvers::lanczos<solvers::policy::TwoPass, Space>(const QKRYLOV_PRECISION_NAMESPACE::MatrixFreeHamiltonian<Space>&, const QKRYLOV_PRECISION_NAMESPACE::LanczosConfig&); \
    template QKRYLOV_PRECISION_NAMESPACE::LanczosLowestResult solvers::lanczos_lowest<Space>(const QKRYLOV_PRECISION_NAMESPACE::MatrixFreeHamiltonian<Space>&, const QKRYLOV_PRECISION_NAMESPACE::LanczosLowestConfig&);

#ifdef KOKKOS_ENABLE_SERIAL
INSTANTIATE_LANCZOS(Kokkos::Serial)
#endif
#ifdef KOKKOS_ENABLE_OPENMP
INSTANTIATE_LANCZOS(Kokkos::OpenMP)
#endif
#ifdef KOKKOS_ENABLE_THREADS
INSTANTIATE_LANCZOS(Kokkos::Threads)
#endif
#ifdef KOKKOS_ENABLE_CUDA
INSTANTIATE_LANCZOS(Kokkos::Cuda)
#endif
#ifdef KOKKOS_ENABLE_HIP
INSTANTIATE_LANCZOS(Kokkos::HIP)
#endif
#ifdef KOKKOS_ENABLE_SYCL
INSTANTIATE_LANCZOS(Kokkos::Experimental::SYCL)
#endif

#undef INSTANTIATE_LANCZOS

} // namespace qkrylov
