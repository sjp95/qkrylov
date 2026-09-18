#pragma once

#include "qkrylov/core/types.hpp"

#include "qkrylov/basis/basis.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"
#include "qkrylov/basis/spin_s_basis.hpp"
#include "qkrylov/basis/fermion_basis.hpp"
#include "qkrylov/basis/hubbard_basis.hpp"
#include "qkrylov/basis/tj_basis.hpp"
#include "qkrylov/sites/site.hpp"
#include "qkrylov/sites/spinhalf_site.hpp"
#include "qkrylov/sites/spin_s_site.hpp"
#include "qkrylov/sites/fermion_site.hpp"
#include "qkrylov/sites/hubbard_site.hpp"
#include "qkrylov/sites/tj_site.hpp"
#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/core/kokkos_types.hpp"
#include "qkrylov/core/device.hpp"
#include "qkrylov/core/traits.hpp"

#include <memory>
#include <vector>
#include <stdexcept>
#include <type_traits>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {

inline std::shared_ptr<Site> infer_site_from_basis(const Basis& basis) {
    if (auto* b = dynamic_cast<const SpinHalfBasis*>(&basis)) {
        return std::make_shared<SpinHalfSite>();
    }
    if (auto* b = dynamic_cast<const SpinSBasis*>(&basis)) {
        return std::make_shared<SpinSSite>(b->spin());
    }
    if (auto* b = dynamic_cast<const FermionBasis*>(&basis)) {
        return std::make_shared<FermionSite>();
    }
    if (auto* b = dynamic_cast<const HubbardBasis*>(&basis)) {
        return std::make_shared<HubbardSite>();
    }
    if (auto* b = dynamic_cast<const TJBasis*>(&basis)) {
        return std::make_shared<TJSite>();
    }
    throw std::invalid_argument("Cannot infer site type from basis. Please provide site explicitly.");
}

/// Matrix-free Hamiltonian with pre-compiled operator action.
///
/// At construction time, the operator sum is evaluated for every basis state
/// to build a CSR (Compressed Sparse Row) representation on the device.
/// This one-time cost eliminates per-apply virtual dispatch, string matching,
/// and hash-map lookups — enabling both GPU execution and faster CPU paths.
template <typename ExecSpace>
class MatrixFreeHamiltonian
{
public:

    MatrixFreeHamiltonian(
        std::shared_ptr<Basis> basis,
        std::shared_ptr<Site> site,
        const OpSum& ops,
        Device device = Device()
    );

    MatrixFreeHamiltonian(
        std::shared_ptr<Basis> basis,
        const OpSum& ops,
        Device device = Device()
    ) : MatrixFreeHamiltonian(basis, infer_site_from_basis(*basis), ops, device) {}

    template <typename BasisType>
        requires std::is_base_of_v<Basis, std::decay_t<BasisType>>
    MatrixFreeHamiltonian(const BasisType& basis, const OpSum& ops, Device device = Device())
        : MatrixFreeHamiltonian(std::make_shared<std::decay_t<BasisType>>(basis), ops, device) {}

    template <typename BasisType>
        requires std::is_base_of_v<Basis, std::decay_t<BasisType>>
    MatrixFreeHamiltonian(const BasisType& basis, std::shared_ptr<Site> site, const OpSum& ops, Device device = Device())
        : MatrixFreeHamiltonian(std::make_shared<std::decay_t<BasisType>>(basis), std::move(site), ops, device) {}

    /// Apply H to a device-resident vector: y = H * x.
    /// No host↔device copies — both x and y must already live on the device.
    void apply(const VectorView<ExecSpace>& x, VectorView<ExecSpace>& y) const;

    /// Apply H to host pointers: y = H * x.
    /// Internally copies host→device, runs the kernel, copies device→host.
    /// This overload exists for the C API and Python binding.
    void apply(const Complex* x, Complex* y) const;

    /// Return the pre-computed diagonal of H as a device-resident vector.
    const VectorView<ExecSpace>& diagonal() const { return diagonal_; }

    /// Return the pre-computed diagonal of H as a host vector.
    HostVector diagonal_host() const;

    /// Hilbert space dimension.
    Index dimension() const { return dim_; }

    /// The device this Hamiltonian was built for.
    const Device& device() const { return device_; }

private:

    Device device_;
    Index dim_ = 0;

    // ---- Pre-compiled CSR representation (device-resident) ----
    // Row-centric storage: row alpha stores all (col, value) pairs
    // where value = conj(H[col][alpha]), enabling a gather-based
    // SpMV with y[alpha] = sum_j values[j] * x[cols[j]].
    // No atomics needed — each thread owns its output element.
    Kokkos::View<Index*, typename ExecSpace::memory_space>    row_offsets_;  // size = dim + 1
    Kokkos::View<Index*, typename ExecSpace::memory_space>    col_indices_;  // size = nnz
    Kokkos::View<KComplex*, typename ExecSpace::memory_space> values_;       // size = nnz

    VectorView<ExecSpace> diagonal_;  // size = dim

    // Cached scratch device views for host apply(const Complex*, Complex*)
    mutable VectorView<ExecSpace> scratch_x_;
    mutable VectorView<ExecSpace> scratch_y_;

    // ---- Original objects (kept for reference/future use) ----
    std::shared_ptr<Basis> basis_;
    std::shared_ptr<Site>  site_;
    OpSum ops_;
};

// CTAD deduction guides for MatrixFreeHamiltonian
template <typename BasisType, typename DeviceTag>
    requires (!std::is_same_v<std::decay_t<DeviceTag>, Device>)
MatrixFreeHamiltonian(const BasisType&, const OpSum&, DeviceTag)
    -> MatrixFreeHamiltonian<typename traits::device_execution_space<std::decay_t<DeviceTag>>::type>;

template <typename BasisType>
MatrixFreeHamiltonian(const BasisType&, const OpSum&)
    -> MatrixFreeHamiltonian<typename traits::device_execution_space<device::cpu>::type>;

template <typename DeviceTag>
    requires (!std::is_same_v<std::decay_t<DeviceTag>, Device>)
MatrixFreeHamiltonian(std::shared_ptr<Basis>, const OpSum&, DeviceTag)
    -> MatrixFreeHamiltonian<typename traits::device_execution_space<std::decay_t<DeviceTag>>::type>;

MatrixFreeHamiltonian(std::shared_ptr<Basis>, const OpSum&)
    -> MatrixFreeHamiltonian<typename traits::device_execution_space<device::cpu>::type>;

template <typename BasisType, typename DeviceTag>
    requires (!std::is_same_v<std::decay_t<DeviceTag>, Device>)
MatrixFreeHamiltonian(const BasisType&, std::shared_ptr<Site>, const OpSum&, DeviceTag)
    -> MatrixFreeHamiltonian<typename traits::device_execution_space<std::decay_t<DeviceTag>>::type>;

template <typename BasisType>
MatrixFreeHamiltonian(const BasisType&, std::shared_ptr<Site>, const OpSum&)
    -> MatrixFreeHamiltonian<typename traits::device_execution_space<device::cpu>::type>;

template <typename DeviceTag>
    requires (!std::is_same_v<std::decay_t<DeviceTag>, Device>)
MatrixFreeHamiltonian(std::shared_ptr<Basis>, std::shared_ptr<Site>, const OpSum&, DeviceTag)
    -> MatrixFreeHamiltonian<typename traits::device_execution_space<std::decay_t<DeviceTag>>::type>;

MatrixFreeHamiltonian(std::shared_ptr<Basis>, std::shared_ptr<Site>, const OpSum&)
    -> MatrixFreeHamiltonian<typename traits::device_execution_space<device::cpu>::type>;

/// Modern alias for MatrixFreeHamiltonian
template <typename ExecSpace = Kokkos::DefaultExecutionSpace>
using Hamiltonian = MatrixFreeHamiltonian<ExecSpace>;

} // namespace QKRYLOV_PRECISION_NAMESPACE

using QKRYLOV_PRECISION_NAMESPACE::Hamiltonian;

} // namespace qkrylov
