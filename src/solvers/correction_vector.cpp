#include "qkrylov/core/types.hpp"
#include "qkrylov/solvers/correction_vector.hpp"
#include "qkrylov/linalg/vector_ops.hpp"

#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <Kokkos_Core.hpp>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {

template <typename ExecSpace>
CorrectionVectorResult correction_vector_spectral(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    const HostVector& Op_psi0,
    Real E0,
    Real omega,
    Real eta,
    int max_iter,
    Real tol
)
{
    const Index dim = H.dimension();
    if (dim == 0) {
        return {HostVector{}, static_cast<Real>(0.0), 0, false};
    }
    if (Op_psi0.size() != dim) {
        throw std::invalid_argument("correction_vector_spectral: Op_psi0 size must match Hamiltonian dimension");
    }

    VectorView<ExecSpace> dev_Op_psi0("qkrylov::cv::Op_psi0", dim);
    copy_host_to_device(Op_psi0, dev_Op_psi0);

    // Right hand side: b = eta * Op_psi0
    VectorView<ExecSpace> b("qkrylov::cv::b", dim);
    deep_copy(b, dev_Op_psi0);
    scal(KComplex(eta, static_cast<Real>(0.0)), b);

    const Real b_norm = norm(b);
    const Real b_min = std::numeric_limits<Real>::epsilon() * Real(4.0);
    if (b_norm < b_min) {
        return {HostVector(dim, Complex(0.0, 0.0)), static_cast<Real>(0.0), 0, true};
    }

    // Allocate workspace for linear operator A
    VectorView<ExecSpace> z("qkrylov::cv::z", dim);
    VectorView<ExecSpace> Ap("qkrylov::cv::Ap", dim);

    const Real shift = E0 + omega;
    const Real eta_sq = eta * eta;

    // Linear operator A y = (H - shift)^2 y + eta^2 y
    auto apply_A = [&](const VectorView<ExecSpace>& y_vec, VectorView<ExecSpace>& Ay_vec) {
        // z = H y - shift * y
        H.apply(y_vec, z);
        axpy(KComplex(-shift, static_cast<Real>(0.0)), y_vec, z);

        // Ay = H z - shift * z + eta^2 * y
        H.apply(z, Ay_vec);
        axpy(KComplex(-shift, static_cast<Real>(0.0)), z, Ay_vec);
        axpy(KComplex(eta_sq, static_cast<Real>(0.0)), y_vec, Ay_vec);
    };

    // Conjugate Gradient (CG) solver for A x = b
    VectorView<ExecSpace> x("qkrylov::cv::x", dim);
    zero_fill(x);

    VectorView<ExecSpace> r("qkrylov::cv::r", dim);
    deep_copy(r, b);

    VectorView<ExecSpace> p("qkrylov::cv::p", dim);
    deep_copy(p, r);

    Real rs_old = dot(r, r).real();

    bool converged = false;
    int iter = 0;
    const Real denom_min = std::numeric_limits<Real>::epsilon() * std::numeric_limits<Real>::epsilon();

    for (iter = 0; iter < max_iter; ++iter) {
        apply_A(p, Ap);

        KComplex p_Ap = dot(p, Ap);
        Real denom = p_Ap.real();
        if (std::abs(denom) < denom_min) {
            break;
        }

        Real alpha = rs_old / denom;

        axpy(KComplex(alpha, static_cast<Real>(0.0)), p, x);
        axpy(KComplex(-alpha, static_cast<Real>(0.0)), Ap, r);

        Real rs_new = dot(r, r).real();
        if (rs_new < static_cast<Real>(0.0)) {
            rs_new = static_cast<Real>(0.0);
        }

        if (std::sqrt(rs_new) / b_norm < tol) {
            converged = true;
            iter++;
            break;
        }

        Real beta_cg = rs_new / rs_old;
        scal(KComplex(beta_cg, static_cast<Real>(0.0)), p);
        axpy(KComplex(1.0, static_cast<Real>(0.0)), r, p);

        rs_old = rs_new;
    }

    // Spectral function S(omega) = (1/pi) * Re<Op_psi0 | x>
    KComplex inner = dot(dev_Op_psi0, x);
    constexpr Real PI = static_cast<Real>(3.141592653589793238462643383279502884L);
    Real spectral_func = (static_cast<Real>(1.0) / PI) * inner.real();

    HostVector host_x;
    copy_device_to_host(x, host_x);

    CorrectionVectorResult result;
    result.correction_vector = std::move(host_x);
    result.spectral_function = spectral_func;
    result.iterations = iter;
    result.converged = converged;

    return result;
}

// Explicit instantiations
#ifdef KOKKOS_ENABLE_SERIAL
template CorrectionVectorResult correction_vector_spectral<Kokkos::Serial>(
    const MatrixFreeHamiltonian<Kokkos::Serial>&,
    const HostVector&,
    Real,
    Real,
    Real,
    int,
    Real
);
#endif

#ifdef KOKKOS_ENABLE_OPENMP
template CorrectionVectorResult correction_vector_spectral<Kokkos::OpenMP>(
    const MatrixFreeHamiltonian<Kokkos::OpenMP>&,
    const HostVector&,
    Real,
    Real,
    Real,
    int,
    Real
);
#endif

#ifdef KOKKOS_ENABLE_THREADS
template CorrectionVectorResult correction_vector_spectral<Kokkos::Threads>(
    const MatrixFreeHamiltonian<Kokkos::Threads>&,
    const HostVector&,
    Real,
    Real,
    Real,
    int,
    Real
);
#endif

#ifdef KOKKOS_ENABLE_CUDA
template CorrectionVectorResult correction_vector_spectral<Kokkos::Cuda>(
    const MatrixFreeHamiltonian<Kokkos::Cuda>&,
    const HostVector&,
    Real,
    Real,
    Real,
    int,
    Real
);
#endif

#ifdef KOKKOS_ENABLE_HIP
template CorrectionVectorResult correction_vector_spectral<Kokkos::HIP>(
    const MatrixFreeHamiltonian<Kokkos::HIP>&,
    const HostVector&,
    Real,
    Real,
    Real,
    int,
    Real
);
#endif

#ifdef KOKKOS_ENABLE_SYCL
template CorrectionVectorResult correction_vector_spectral<Kokkos::Experimental::SYCL>(
    const MatrixFreeHamiltonian<Kokkos::Experimental::SYCL>&,
    const HostVector&,
    Real,
    Real,
    Real,
    int,
    Real
);
#endif

} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov
