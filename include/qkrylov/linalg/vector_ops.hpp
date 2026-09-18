#pragma once

#include "qkrylov/core/types.hpp"
#include "qkrylov/core/kokkos_types.hpp"

#include <cmath>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {



/// Compute the inner product <x|y> = sum_i conj(x_i) * y_i
template <typename ViewType>
inline KComplex dot(
    const ViewType& x,
    const ViewType& y
)
{
    using ExecSpace = typename ViewType::execution_space;
    KComplex result(0.0, 0.0);

    Kokkos::parallel_reduce("qkrylov::dot",
        Kokkos::RangePolicy<ExecSpace, Index>(0, x.extent(0)),
        KOKKOS_LAMBDA(const Index i, KComplex& sum) {
            sum += Kokkos::conj(x(i)) * y(i);
        },
        result
    );

    return result;
}

/// Compute the 2-norm ||x||
template <typename ViewType>
inline Real norm(
    const ViewType& x
)
{
    return std::sqrt(
        dot(x, x).real()
    );
}

/// Scale: x = a * x
template <typename ViewType>
inline void scal(
    KComplex a,
    ViewType& x
)
{
    using ExecSpace = typename ViewType::execution_space;
    Kokkos::parallel_for("qkrylov::scal",
        Kokkos::RangePolicy<ExecSpace, Index>(0, x.extent(0)),
        KOKKOS_LAMBDA(const Index i) {
            x(i) *= a;
        }
    );
}

/// Scale: x = a * x  (real scalar convenience overload)
template <typename ViewType>
inline void scal(
    Real a,
    ViewType& x
)
{
    scal(KComplex(a, 0.0), x);
}

/// AXPY: y = a*x + y
template <typename ViewType>
inline void axpy(
    KComplex a,
    const ViewType& x,
    ViewType& y
)
{
    using ExecSpace = typename ViewType::execution_space;
    Kokkos::parallel_for("qkrylov::axpy",
        Kokkos::RangePolicy<ExecSpace, Index>(0, x.extent(0)),
        KOKKOS_LAMBDA(const Index i) {
            y(i) += a * x(i);
        }
    );
}

/// Normalize x to unit length (in-place).
template <typename ViewType>
inline void normalize(
    ViewType& x
)
{
    const Real n = norm(x);
    if (n == 0.0) return;
    scal(KComplex(1.0 / n, 0.0), x);
}

/// Fill a device vector with zeros.
template <typename ViewType>
inline void zero_fill(ViewType& x) {
    using ExecSpace = typename ViewType::execution_space;
    Kokkos::deep_copy(ExecSpace(), x, KComplex(0.0, 0.0));
}

/// Deep-copy src into dst (same layout, same extent).
template <typename ViewType1, typename ViewType2>
inline void deep_copy(ViewType1& dst, const ViewType2& src) {
    using ExecSpace = typename ViewType1::execution_space;
    Kokkos::deep_copy(ExecSpace(), dst, src);
}

/// Copy a HostVector (std::vector<Complex>) into a device Vector.
/// The device view must already be allocated with the correct size.
template <typename ViewType>
inline void copy_host_to_device(const HostVector& host, ViewType& device) {
    using ExecSpace = typename ViewType::execution_space;
    auto h_view = Kokkos::View<const KComplex*, Kokkos::HostSpace,
                               Kokkos::MemoryUnmanaged>(
        reinterpret_cast<const KComplex*>(host.data()),
        host.size()
    );
    Kokkos::deep_copy(ExecSpace(), device, h_view);
}

/// Copy a device Vector into a HostVector (std::vector<Complex>).
/// Resizes the host vector to match.
template <typename ViewType>
inline void copy_device_to_host(const ViewType& device, HostVector& host) {
    host.resize(device.extent(0));
    auto h_view = Kokkos::View<KComplex*, Kokkos::HostSpace,
                               Kokkos::MemoryUnmanaged>(
        reinterpret_cast<KComplex*>(host.data()),
        host.size()
    );
    Kokkos::deep_copy(h_view, device);
}



} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov
