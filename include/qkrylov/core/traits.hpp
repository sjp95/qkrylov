#pragma once

#include "qkrylov/core/device.hpp"
#include <Kokkos_Core.hpp>

namespace qkrylov {
namespace traits {

template <typename Tag>
struct device_execution_space;

template <>
struct device_execution_space<device::cpu> {
#if defined(KOKKOS_ENABLE_OPENMP)
    using type = Kokkos::OpenMP;
#elif defined(KOKKOS_ENABLE_THREADS)
    using type = Kokkos::Threads;
#else
    using type = Kokkos::Serial;
#endif
};

template <>
struct device_execution_space<device::gpu> {
#if defined(KOKKOS_ENABLE_CUDA)
    using type = Kokkos::Cuda;
#elif defined(KOKKOS_ENABLE_HIP)
    using type = Kokkos::HIP;
#elif defined(KOKKOS_ENABLE_SYCL)
    using type = Kokkos::Experimental::SYCL;
#else
    using type = typename device_execution_space<device::cpu>::type;
#endif
};

template <>
struct device_execution_space<device::openmp> {
#if defined(KOKKOS_ENABLE_OPENMP)
    using type = Kokkos::OpenMP;
#else
    using type = Kokkos::DefaultHostExecutionSpace;
#endif
};

template <>
struct device_execution_space<device::serial> {
    using type = Kokkos::Serial;
};

template <>
struct device_execution_space<device::cuda> {
#if defined(KOKKOS_ENABLE_CUDA)
    using type = Kokkos::Cuda;
#else
    using type = typename device_execution_space<device::cpu>::type;
#endif
};

template <>
struct device_execution_space<device::hip> {
#if defined(KOKKOS_ENABLE_HIP)
    using type = Kokkos::HIP;
#else
    using type = typename device_execution_space<device::cpu>::type;
#endif
};

template <>
struct device_execution_space<device::sycl> {
#if defined(KOKKOS_ENABLE_SYCL)
    using type = Kokkos::Experimental::SYCL;
#else
    using type = typename device_execution_space<device::cpu>::type;
#endif
};

template <typename Tag>
using device_execution_space_t = typename device_execution_space<Tag>::type;

template <typename Tag>
struct device_traits {
    using execution_space = typename device_execution_space<Tag>::type;
};

} // namespace traits
} // namespace qkrylov
