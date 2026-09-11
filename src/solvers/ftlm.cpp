#include "qkrylov/core/types.hpp"
#include "qkrylov/solvers/ftlm.hpp"
#include "qkrylov/linalg/vector_ops.hpp"

#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <Kokkos_Core.hpp>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {

namespace
{

struct FullTridiagResult {
    std::vector<Real> eigenvalues;                       // size M
    std::vector<Real> first_components;                  // size M
    std::vector<std::vector<Real>> ritz_vectors;          // z[k][m]: k-th component of m-th eigenvector
};

FullTridiagResult diagonalize_tridiag_full(const std::vector<Real>& alpha, const std::vector<Real>& beta)
{
    int n = alpha.size();
    if (n == 0) return {};
    if (n == 1) return { {alpha[0]}, {1.0}, {{1.0}} };

    std::vector<Real> d = alpha;
    std::vector<Real> e = beta;
    std::vector<std::vector<Real>> z(n, std::vector<Real>(n, 0.0));
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

    FullTridiagResult res;
    res.eigenvalues = d;
    res.first_components.resize(n);
    for (int m = 0; m < n; ++m) {
        res.first_components[m] = z[0][m];
    }
    res.ritz_vectors = z;
    return res;
}

} // anonymous namespace

// -----------------------------------------------------------------------------
// FTLMResult Methods
// -----------------------------------------------------------------------------

template <typename ExecSpace>
Real FTLMResult<ExecSpace>::partition_function(Real b) const
{
    if (samples.empty()) return 0.0;
    Real sum_z = 0.0;
    for (const auto& sample : samples) {
        Real sample_z = 0.0;
        for (size_t m = 0; m < sample.eigenvalues.size(); ++m) {
            Real w_m = (sample.norm_r * sample.norm_r) * (sample.first_components[m] * sample.first_components[m]) * std::exp(-b * sample.eigenvalues[m]);
            sample_z += w_m;
        }
        sum_z += sample_z;
    }
    return (sum_z / samples.size()) * (hilbert_dim > 0 ? hilbert_dim / static_cast<Real>(hilbert_dim) : 1.0);
}

template <typename ExecSpace>
std::vector<Real> FTLMResult<ExecSpace>::partition_function(const std::vector<Real>& betas) const
{
    std::vector<Real> res(betas.size());
    for (size_t i = 0; i < betas.size(); ++i) {
        res[i] = partition_function(betas[i]);
    }
    return res;
}

template <typename ExecSpace>
Real FTLMResult<ExecSpace>::internal_energy(Real b) const
{
    if (samples.empty()) return 0.0;
    Real sum_z = 0.0;
    Real sum_e = 0.0;
    for (const auto& sample : samples) {
        for (size_t m = 0; m < sample.eigenvalues.size(); ++m) {
            Real w_m = (sample.norm_r * sample.norm_r) * (sample.first_components[m] * sample.first_components[m]) * std::exp(-b * sample.eigenvalues[m]);
            sum_z += w_m;
            sum_e += sample.eigenvalues[m] * w_m;
        }
    }
    return (sum_z > 0.0) ? (sum_e / sum_z) : 0.0;
}

template <typename ExecSpace>
std::vector<Real> FTLMResult<ExecSpace>::internal_energy(const std::vector<Real>& betas) const
{
    std::vector<Real> res(betas.size());
    for (size_t i = 0; i < betas.size(); ++i) {
        res[i] = internal_energy(betas[i]);
    }
    return res;
}

template <typename ExecSpace>
Real FTLMResult<ExecSpace>::expectation_value(const MatrixFreeHamiltonian<ExecSpace>& A, Real b) const
{
    if (samples.empty()) return 0.0;
    Real sum_z = 0.0;
    Real sum_a = 0.0;

    const Index dim = A.dimension();

    for (const auto& sample : samples) {
        int M = sample.krylov_basis.size();
        if (M == 0) continue;

        // Build A_krylov matrix (M x M) where A_ij = <v_i | A | v_j>
        std::vector<std::vector<Complex>> A_krylov(M, std::vector<Complex>(M, 0.0));
        VectorView<ExecSpace> w("w_A", dim);

        for (int j = 0; j < M; ++j) {
            A.apply(sample.krylov_basis[j], w);
            for (int i = 0; i < M; ++i) {
                A_krylov[i][j] = dot(sample.krylov_basis[i], w);
            }
        }

        // Evaluate FTLM matrix elements for sample r:
        // <r | y_m> = norm_r * Y_{0,m}
        // <y_m | A | r> = norm_r * sum_i Y_{i,m} A_krylov[i][0]
        for (int m = 0; m < M; ++m) {
            Real c_r_ym = sample.norm_r * sample.first_components[m];
            Complex ym_A_v0 = 0.0;
            for (int i = 0; i < M; ++i) {
                Real y_im = sample.ritz_vectors[i][m];
                ym_A_v0 += y_im * A_krylov[i][0];
            }
            Real ym_A_r = sample.norm_r * ym_A_v0.real();

            Real boltzmann = std::exp(-b * sample.eigenvalues[m]);
            Real w_m = c_r_ym * c_r_ym * boltzmann;
            sum_z += w_m;
            sum_a += c_r_ym * ym_A_r * boltzmann;
        }
    }

    return (sum_z > 0.0) ? (sum_a / sum_z) : 0.0;
}

template <typename ExecSpace>
std::vector<Real> FTLMResult<ExecSpace>::expectation_value(const MatrixFreeHamiltonian<ExecSpace>& A, const std::vector<Real>& betas) const
{
    std::vector<Real> res(betas.size());
    for (size_t i = 0; i < betas.size(); ++i) {
        res[i] = expectation_value(A, betas[i]);
    }
    return res;
}

// -----------------------------------------------------------------------------
// ftlm Solver
// -----------------------------------------------------------------------------

template <typename ExecSpace>
FTLMResult<ExecSpace> ftlm(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    Real beta,
    int n_random,
    int n_steps
)
{
    const Index dim = H.dimension();
    FTLMResult<ExecSpace> res;
    res.beta = beta;
    res.hilbert_dim = dim;
    if (dim == 0) return res;

    std::mt19937 rng(42);
    // Standard complex Gaussian vector: Real & Imag parts mean 0, stddev 1/sqrt(2)
    std::normal_distribution<Real> dist(0.0, 1.0 / std::sqrt(2.0));

    for (int r = 0; r < n_random; ++r) {
        FTLMSample<ExecSpace> sample;

        VectorView<ExecSpace> r_vec("r_vec", dim);
        auto r_vec_host = Kokkos::create_mirror_view(r_vec);
        for (Index i = 0; i < dim; ++i) {
            r_vec_host(i) = KComplex(dist(rng), dist(rng));
        }
        Kokkos::deep_copy(r_vec, r_vec_host);

        Real nrm = norm(r_vec);
        sample.norm_r = nrm;

        if (nrm < 1e-15) continue;

        // Lanczos loop storing Krylov basis
        VectorView<ExecSpace> v_curr("v_curr", dim);
        Kokkos::deep_copy(v_curr, r_vec);
        scal(1.0 / nrm, v_curr);

        VectorView<ExecSpace> v_prev("v_prev", dim);
        VectorView<ExecSpace> w("w", dim);

        std::vector<Real> alphas;
        std::vector<Real> betas_l;

        for (int i = 0; i < n_steps; ++i) {
            VectorView<ExecSpace> v_copy("v_k", dim);
            Kokkos::deep_copy(v_copy, v_curr);
            sample.krylov_basis.push_back(v_copy);

            H.apply(v_curr, w);
            Real alpha = dot(v_curr, w).real();
            alphas.push_back(alpha);

            axpy(-alpha, v_curr, w);
            if (i > 0) {
                axpy(-betas_l.back(), v_prev, w);
            }

            Real b_val = norm(w);
            if (b_val < 1e-15) break;

            betas_l.push_back(b_val);
            Kokkos::deep_copy(v_prev, v_curr);
            Kokkos::deep_copy(v_curr, w);
            scal(1.0 / b_val, v_curr);
        }

        auto tridiag_res = diagonalize_tridiag_full(alphas, betas_l);
        sample.eigenvalues = tridiag_res.eigenvalues;
        sample.first_components = tridiag_res.first_components;
        sample.ritz_vectors = tridiag_res.ritz_vectors;

        res.samples.push_back(sample);
    }

    res.partition_function_val = res.partition_function(beta);
    res.internal_energy_val = res.internal_energy(beta);

    return res;
}

// -----------------------------------------------------------------------------
// ftlm_dynamical_correlation Solver
// -----------------------------------------------------------------------------

template <typename ExecSpace>
std::vector<Real> ftlm_dynamical_correlation(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    const MatrixFreeHamiltonian<ExecSpace>& A,
    const MatrixFreeHamiltonian<ExecSpace>& B,
    Real beta,
    int n_random,
    int n_steps_thermal,
    int n_steps_dyn,
    const std::vector<Real>& omegas,
    Real eta
)
{
    const Index dim = H.dimension();
    std::vector<Real> spectral(omegas.size(), 0.0);
    if (dim == 0 || omegas.empty()) return spectral;

    std::mt19937 rng(42);
    std::normal_distribution<Real> dist(0.0, 1.0 / std::sqrt(2.0));

    Real total_Z = 0.0;

    for (int r = 0; r < n_random; ++r) {
        VectorView<ExecSpace> r_vec("r_vec", dim);
        auto r_vec_host = Kokkos::create_mirror_view(r_vec);
        for (Index i = 0; i < dim; ++i) {
            r_vec_host(i) = KComplex(dist(rng), dist(rng));
        }
        Kokkos::deep_copy(r_vec, r_vec_host);

        Real norm_r = norm(r_vec);
        if (norm_r < 1e-15) continue;

        // Step 1: Thermalization Lanczos on H
        VectorView<ExecSpace> v_curr("v_curr", dim);
        Kokkos::deep_copy(v_curr, r_vec);
        scal(1.0 / norm_r, v_curr);

        VectorView<ExecSpace> v_prev("v_prev", dim);
        VectorView<ExecSpace> w("w", dim);

        std::vector<VectorView<ExecSpace>> krylov_1;
        std::vector<Real> alphas_1, betas_1;

        for (int i = 0; i < n_steps_thermal; ++i) {
            VectorView<ExecSpace> v_copy("v_k1", dim);
            Kokkos::deep_copy(v_copy, v_curr);
            krylov_1.push_back(v_copy);

            H.apply(v_curr, w);
            Real alpha = dot(v_curr, w).real();
            alphas_1.push_back(alpha);

            axpy(-alpha, v_curr, w);
            if (i > 0) axpy(-betas_1.back(), v_prev, w);

            Real b_val = norm(w);
            if (b_val < 1e-15) break;

            betas_1.push_back(b_val);
            Kokkos::deep_copy(v_prev, v_curr);
            Kokkos::deep_copy(v_curr, w);
            scal(1.0 / b_val, v_curr);
        }

        auto tridiag_1 = diagonalize_tridiag_full(alphas_1, betas_1);
        int M1 = krylov_1.size();

        // Accumulate sample Z
        Real sample_Z = 0.0;
        for (int m = 0; m < M1; ++m) {
            Real w_m = (norm_r * norm_r) * (tridiag_1.first_components[m] * tridiag_1.first_components[m]) * std::exp(-beta * tridiag_1.eigenvalues[m]);
            sample_Z += w_m;
        }
        total_Z += sample_Z;

        // For each Ritz eigenvector |y_m> in thermal Krylov space:
        for (int m = 0; m < M1; ++m) {
            Real E_m = tridiag_1.eigenvalues[m];
            Real c_r_ym = norm_r * tridiag_1.first_components[m]; // <r | y_m>

            // Reconstruct full Ritz vector |y_m> = sum_k Y_km |v_k>
            VectorView<ExecSpace> y_m("y_m", dim);
            zero_fill(y_m);
            for (int k = 0; k < M1; ++k) {
                axpy(Complex(tridiag_1.ritz_vectors[k][m], 0.0), krylov_1[k], y_m);
            }

            // Compute seed state |phi_m> = B |y_m>
            VectorView<ExecSpace> phi_m("phi_m", dim);
            B.apply(y_m, phi_m);
            Real norm_phi = norm(phi_m);
            if (norm_phi < 1e-15) continue;

            // Step 2: Second Lanczos run on H starting from |phi_m> / norm_phi
            VectorView<ExecSpace> v2_curr("v2_curr", dim);
            Kokkos::deep_copy(v2_curr, phi_m);
            scal(1.0 / norm_phi, v2_curr);

            VectorView<ExecSpace> v2_prev("v2_prev", dim);
            VectorView<ExecSpace> w2("w2", dim);

            std::vector<VectorView<ExecSpace>> krylov_2;
            std::vector<Real> alphas_2, betas_2;

            for (int i = 0; i < n_steps_dyn; ++i) {
                VectorView<ExecSpace> v_copy2("v_k2", dim);
                Kokkos::deep_copy(v_copy2, v2_curr);
                krylov_2.push_back(v_copy2);

                H.apply(v2_curr, w2);
                Real alpha = dot(v2_curr, w2).real();
                alphas_2.push_back(alpha);

                axpy(-alpha, v2_curr, w2);
                if (i > 0) axpy(-betas_2.back(), v2_prev, w2);

                Real b_val = norm(w2);
                if (b_val < 1e-15) break;

                betas_2.push_back(b_val);
                Kokkos::deep_copy(v2_prev, v2_curr);
                Kokkos::deep_copy(v2_curr, w2);
                scal(1.0 / b_val, v2_curr);
            }

            auto tridiag_2 = diagonalize_tridiag_full(alphas_2, betas_2);
            int M2 = krylov_2.size();

            // Evaluate matrix elements <z_p | A | y_m>
            // Compute A |y_m>
            VectorView<ExecSpace> A_ym("A_ym", dim);
            A.apply(y_m, A_ym);

            for (int p = 0; p < M2; ++p) {
                Real E_tilde_p = tridiag_2.eigenvalues[p];

                // Reconstruct Ritz state |z_p> = sum_j Z_jp |v2_j>
                // We want <z_p | A | y_m> = sum_j Z_jp <v2_j | A | y_m>
                Complex z_p_A_ym = 0.0;
                for (int j = 0; j < M2; ++j) {
                    Real Z_jp = tridiag_2.ritz_vectors[j][p];
                    Complex dot_val = dot(krylov_2[j], A_ym);
                    z_p_A_ym += Z_jp * dot_val;
                }

                // Weight factor: exp(-beta * E_m) * <r | y_m>^2 * <y_m | B^\dagger | z_p> <z_p | A | y_m>
                // Note: <z_p | B | y_m> = norm_phi * Z_0p (since |phi_m> = B|y_m> = norm_phi * |v2_0>)
                // Therefore <y_m | B^\dagger | z_p> = norm_phi * Z_0p
                Real weight = std::exp(-beta * E_m) * (c_r_ym * c_r_ym) * norm_phi * tridiag_2.first_components[p] * z_p_A_ym.real();

                Real dE = E_tilde_p - E_m;

                // Accumulate Lorentzian/Gaussian broadened delta functions
                for (size_t i = 0; i < omegas.size(); ++i) {
                    Real w_val = omegas[i];
                    Real diff = w_val - dE;
                    Real delta = (1.0 / M_PI) * (eta / (diff * diff + eta * eta));
                    spectral[i] += weight * delta;
                }
            }
        }
    }

    if (total_Z > 0.0) {
        for (size_t i = 0; i < omegas.size(); ++i) {
            spectral[i] /= total_Z;
        }
    }

    return spectral;
}

// Explicit instantiations
#ifdef KOKKOS_ENABLE_SERIAL
template struct FTLMResult<Kokkos::Serial>;
template FTLMResult<Kokkos::Serial> ftlm<Kokkos::Serial>(const MatrixFreeHamiltonian<Kokkos::Serial>&, Real, int, int);
template std::vector<Real> ftlm_dynamical_correlation<Kokkos::Serial>(
    const MatrixFreeHamiltonian<Kokkos::Serial>&, const MatrixFreeHamiltonian<Kokkos::Serial>&, const MatrixFreeHamiltonian<Kokkos::Serial>&,
    Real, int, int, int, const std::vector<Real>&, Real);
#endif

#ifdef KOKKOS_ENABLE_OPENMP
template struct FTLMResult<Kokkos::OpenMP>;
template FTLMResult<Kokkos::OpenMP> ftlm<Kokkos::OpenMP>(const MatrixFreeHamiltonian<Kokkos::OpenMP>&, Real, int, int);
template std::vector<Real> ftlm_dynamical_correlation<Kokkos::OpenMP>(
    const MatrixFreeHamiltonian<Kokkos::OpenMP>&, const MatrixFreeHamiltonian<Kokkos::OpenMP>&, const MatrixFreeHamiltonian<Kokkos::OpenMP>&,
    Real, int, int, int, const std::vector<Real>&, Real);
#endif

#ifdef KOKKOS_ENABLE_THREADS
template struct FTLMResult<Kokkos::Threads>;
template FTLMResult<Kokkos::Threads> ftlm<Kokkos::Threads>(const MatrixFreeHamiltonian<Kokkos::Threads>&, Real, int, int);
template std::vector<Real> ftlm_dynamical_correlation<Kokkos::Threads>(
    const MatrixFreeHamiltonian<Kokkos::Threads>&, const MatrixFreeHamiltonian<Kokkos::Threads>&, const MatrixFreeHamiltonian<Kokkos::Threads>&,
    Real, int, int, int, const std::vector<Real>&, Real);
#endif

#ifdef KOKKOS_ENABLE_CUDA
template struct FTLMResult<Kokkos::Cuda>;
template FTLMResult<Kokkos::Cuda> ftlm<Kokkos::Cuda>(const MatrixFreeHamiltonian<Kokkos::Cuda>&, Real, int, int);
template std::vector<Real> ftlm_dynamical_correlation<Kokkos::Cuda>(
    const MatrixFreeHamiltonian<Kokkos::Cuda>&, const MatrixFreeHamiltonian<Kokkos::Cuda>&, const MatrixFreeHamiltonian<Kokkos::Cuda>&,
    Real, int, int, int, const std::vector<Real>&, Real);
#endif

#ifdef KOKKOS_ENABLE_HIP
template struct FTLMResult<Kokkos::HIP>;
template FTLMResult<Kokkos::HIP> ftlm<Kokkos::HIP>(const MatrixFreeHamiltonian<Kokkos::HIP>&, Real, int, int);
template std::vector<Real> ftlm_dynamical_correlation<Kokkos::HIP>(
    const MatrixFreeHamiltonian<Kokkos::HIP>&, const MatrixFreeHamiltonian<Kokkos::HIP>&, const MatrixFreeHamiltonian<Kokkos::HIP>&,
    Real, int, int, int, const std::vector<Real>&, Real);
#endif

#ifdef KOKKOS_ENABLE_SYCL
template struct FTLMResult<Kokkos::Experimental::SYCL>;
template FTLMResult<Kokkos::Experimental::SYCL> ftlm<Kokkos::Experimental::SYCL>(const MatrixFreeHamiltonian<Kokkos::Experimental::SYCL>&, Real, int, int);
template std::vector<Real> ftlm_dynamical_correlation<Kokkos::Experimental::SYCL>(
    const MatrixFreeHamiltonian<Kokkos::Experimental::SYCL>&, const MatrixFreeHamiltonian<Kokkos::Experimental::SYCL>&, const MatrixFreeHamiltonian<Kokkos::Experimental::SYCL>&,
    Real, int, int, int, const std::vector<Real>&, Real);
#endif

} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov
