#include "qkrylov/core/types.hpp"
#include "qkrylov/solvers/dynamics.hpp"

#include <cmath>
#include <complex>
#include <limits>
#include <Kokkos_Core.hpp>

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
    if (dim == 0) return {};

    VectorView<ExecSpace> dev_phi0("dev_phi0", dim);
    copy_host_to_device(phi0, dev_phi0);

    const Real mach_eps = std::numeric_limits<Real>::epsilon() * Real(4.0);
    Real norm_phi = norm(dev_phi0);
    if (norm_phi < mach_eps) return { {}, {}, 0.0 };

    VectorView<ExecSpace> v_curr("v_curr", dim);
    Kokkos::deep_copy(v_curr, dev_phi0);
    scal(1.0/norm_phi, v_curr);

    VectorView<ExecSpace> v_prev("v_prev", dim);
    VectorView<ExecSpace> w("w", dim);

    DynamicsResult res;
    res.norm_phi0 = norm_phi;

    for (int iter = 0; iter < n_iter; ++iter) {
        H.apply(v_curr, w);

        Real alpha = dot(v_curr, w).real();
        res.alphas.push_back(alpha);

        axpy(-alpha, v_curr, w);
        if (iter > 0) {
            axpy(-res.betas.back(), v_prev, w);
        }

        Real beta = norm(w);
        if (beta < mach_eps) break;

        res.betas.push_back(beta);
        Kokkos::deep_copy(v_prev, v_curr);
        Kokkos::deep_copy(v_curr, w);
        scal(1.0/beta, v_curr);
    }

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
