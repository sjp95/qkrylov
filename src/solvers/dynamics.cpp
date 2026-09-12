#include "qkrylov/core/types.hpp"
#include "qkrylov/solvers/dynamics.hpp"
#include "qkrylov/solvers/lanczos.hpp"

#include <cmath>
#include <complex>
#include <limits>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {

template <typename ExecSpace>
DynamicsResult continued_fraction_coeffs(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    const HostVector& phi0,
    int n_iter
)
{
    const Index dim = H.dimension();
    if (dim == 0 || phi0.empty()) return {};

    Real norm_sq = Real(0.0);
    for (Index i = 0; i < phi0.size(); ++i) {
        norm_sq += std::norm(phi0[i]);
    }
    const Real norm_phi = std::sqrt(norm_sq);

    const Real mach_eps = std::numeric_limits<Real>::epsilon() * Real(4.0);
    if (norm_phi < mach_eps) return { {}, {}, Real(0.0) };

    LanczosConfig cfg;
    cfg.maxiter = n_iter;
    cfg.min_iterations = n_iter;
    cfg.tol = Real(0.0);
    cfg.initial_vector = phi0;
    cfg.compute_eigenvectors = false;

    auto l_res = solvers::lanczos<solvers::policy::OnePass>(H, cfg);

    DynamicsResult res;
    res.alphas = std::move(l_res.alphas);
    res.betas = std::move(l_res.betas);
    res.norm_phi0 = norm_phi;
    return res;
}

Real evaluate_spectral_function(
    const Real* alphas,
    const Real* betas,
    size_t n,
    Real norm_phi0,
    Real omega,
    Real E0,
    Real eta
)
{
    if (n == 0) return 0.0;

    std::complex<Real> z(omega + E0, eta);

    // Backward recursion for continued fraction
    // C_n = 1 / (z - a_n)
    // C_{i} = 1 / (z - a_i - b_i^2 * C_{i+1})

    std::complex<Real> f = 0.0;
    for (int i = n - 1; i >= 0; --i) {
        if (i == n - 1) {
            f = Real(1.0) / (z - alphas[i]);
        } else {
            f = Real(1.0) / (z - alphas[i] - betas[i] * betas[i] * f);
        }
    }

    constexpr Real PI = static_cast<Real>(3.141592653589793238462643383279502884L);
    return -static_cast<Real>(1.0) / PI * std::imag(norm_phi0 * norm_phi0 * f);
}


// Explicit instantiations
#ifdef KOKKOS_ENABLE_SERIAL
template DynamicsResult continued_fraction_coeffs<Kokkos::Serial>(const MatrixFreeHamiltonian<Kokkos::Serial>&, const HostVector&, int);
#endif
#ifdef KOKKOS_ENABLE_OPENMP
template DynamicsResult continued_fraction_coeffs<Kokkos::OpenMP>(const MatrixFreeHamiltonian<Kokkos::OpenMP>&, const HostVector&, int);
#endif
#ifdef KOKKOS_ENABLE_THREADS
template DynamicsResult continued_fraction_coeffs<Kokkos::Threads>(const MatrixFreeHamiltonian<Kokkos::Threads>&, const HostVector&, int);
#endif
#ifdef KOKKOS_ENABLE_CUDA
template DynamicsResult continued_fraction_coeffs<Kokkos::Cuda>(const MatrixFreeHamiltonian<Kokkos::Cuda>&, const HostVector&, int);
#endif
#ifdef KOKKOS_ENABLE_HIP
template DynamicsResult continued_fraction_coeffs<Kokkos::HIP>(const MatrixFreeHamiltonian<Kokkos::HIP>&, const HostVector&, int);
#endif
#ifdef KOKKOS_ENABLE_SYCL
template DynamicsResult continued_fraction_coeffs<Kokkos::Experimental::SYCL>(const MatrixFreeHamiltonian<Kokkos::Experimental::SYCL>&, const HostVector&, int);
#endif
}

}
