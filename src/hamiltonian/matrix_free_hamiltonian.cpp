#include "qkrylov/core/types.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"
#include "qkrylov/linalg/vector_ops.hpp"

#include <stdexcept>
#include <vector>
#include <algorithm>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {


template <typename ExecSpace>
MatrixFreeHamiltonian<ExecSpace>::MatrixFreeHamiltonian(
    std::shared_ptr<Basis> basis,
    std::shared_ptr<Site> site,
    const OpSum& ops,
    Device device
)
    : device_(device),
      basis_(std::move(basis)),
      site_(std::move(site)),
      ops_(ops)
{
    detail::initialize_kokkos(device_);

    dim_ = basis_->size();
    const Index dim = dim_;

    // ----------------------------------------------------------------
    // Phase 1: Build the CSR on host using the existing Basis / Site
    //          virtual interfaces.  This runs once at construction time
    //          and eliminates all virtual dispatch, string matching, and
    //          hash-map lookups from the per-apply hot path.
    // ----------------------------------------------------------------

    std::vector<Index>   h_row_offsets(dim + 1);
    std::vector<Index>   h_col_indices;
    std::vector<KComplex> h_values;
    std::vector<KComplex> h_diagonal(dim, KComplex(0.0, 0.0));

    // Conservative reservation (most terms produce one entry per state).
    h_col_indices.reserve(dim * ops_.size());
    h_values.reserve(dim * ops_.size());

    std::vector<std::pair<Index, KComplex>> row_entries;
    row_entries.reserve(ops_.size());

    for (Index alpha = 0; alpha < dim; ++alpha) {
        h_row_offsets[alpha] = h_col_indices.size();
        row_entries.clear();
        const StateID initial_state = basis_->state(alpha);

        for (const auto& term : ops_.terms()) {
            StateID state  = initial_state;
            Complex amp    = term.coeff;
            bool    valid  = true;

            for (const auto& factor : term.factors) {
                auto action = site_->apply(
                    factor.op,
                    factor.site,
                    state
                );

                if (!action.valid) {
                    valid = false;
                    break;
                }

                state = action.new_state;
                amp  *= action.matrix_element;
            }

            if (!valid)                    continue;
            if (!basis_->contains(state))  continue;

            const Index beta = basis_->index(state);

            // The source-based iteration gives  H[beta][alpha] = amp.
            // By Hermiticity:  H[alpha][beta] = conj(amp).
            // We store (col = beta, value = conj(amp)) in row alpha
            // so the gather kernel computes:
            //   y[alpha] = sum_j values[j] * x[cols[j]]
            // with no atomics (each thread writes only to its own y element).
            KComplex val(amp.real(), -amp.imag());  // conj(amp)

            row_entries.emplace_back(beta, val);
        }

        if (!row_entries.empty()) {
            std::sort(row_entries.begin(), row_entries.end(),
                [](const auto& a, const auto& b) {
                    return a.first < b.first;
                });

            for (size_t k = 0; k < row_entries.size(); ) {
                Index col = row_entries[k].first;
                KComplex sum_val = row_entries[k].second;
                size_t next = k + 1;
                while (next < row_entries.size() && row_entries[next].first == col) {
                    sum_val += row_entries[next].second;
                    ++next;
                }

                h_col_indices.push_back(col);
                h_values.push_back(sum_val);

                // Accumulate diagonal
                if (col == alpha) {
                    h_diagonal[alpha] = sum_val;
                }
                k = next;
            }
        }
    }
    h_row_offsets[dim] = h_col_indices.size();

    const Index nnz = h_col_indices.size();

    // ----------------------------------------------------------------
    // Phase 2: Deep-copy the CSR arrays to device memory.
    // ----------------------------------------------------------------

    using MemSpace = typename ExecSpace::memory_space;

    row_offsets_ = Kokkos::View<Index*, MemSpace>("qkrylov::row_offsets", dim + 1);
    col_indices_ = Kokkos::View<Index*, MemSpace>("qkrylov::col_indices", nnz);
    values_      = Kokkos::View<KComplex*, MemSpace>("qkrylov::values", nnz);
    diagonal_    = VectorView<ExecSpace>("qkrylov::diagonal", dim);

    // Wrap host std::vectors as unmanaged Kokkos HostSpace views, then
    // deep_copy into the device views.
    {
        auto h_ro = Kokkos::View<const Index*,
                                 Kokkos::HostSpace,
                                 Kokkos::MemoryUnmanaged>(
            h_row_offsets.data(), dim + 1);

        auto h_ci = Kokkos::View<const Index*,
                                 Kokkos::HostSpace,
                                 Kokkos::MemoryUnmanaged>(
            h_col_indices.data(), nnz);

        auto h_vl = Kokkos::View<const KComplex*,
                                 Kokkos::HostSpace,
                                 Kokkos::MemoryUnmanaged>(
            h_values.data(), nnz);

        auto h_dg = Kokkos::View<const KComplex*,
                                 Kokkos::HostSpace,
                                 Kokkos::MemoryUnmanaged>(
            h_diagonal.data(), dim);

        Kokkos::deep_copy(ExecSpace(), row_offsets_, h_ro);
        Kokkos::deep_copy(ExecSpace(), col_indices_, h_ci);
        Kokkos::deep_copy(ExecSpace(), values_,      h_vl);
        Kokkos::deep_copy(ExecSpace(), diagonal_,    h_dg);
    }
}

// ====================================================================
//  Device-view overload (zero-copy, used by solvers internally)
// ====================================================================
template <typename ExecSpace>
void MatrixFreeHamiltonian<ExecSpace>::apply(
    const VectorView<ExecSpace>& x,
    VectorView<ExecSpace>& y
) const
{
    const Index dim = dim_;
    auto rows = row_offsets_;
    auto cols = col_indices_;
    auto vals = values_;

    // Gather-based SpMV:  y[alpha] = sum_j vals[j] * x[cols[j]]
    // Each thread owns its y[alpha] — no atomics needed.
    Kokkos::parallel_for("qkrylov::H_apply",
        Kokkos::RangePolicy<ExecSpace, Index>(0, dim),
        KOKKOS_LAMBDA(const Index alpha) {
            KComplex sum(0.0, 0.0);
            const Index row_begin = rows(alpha);
            const Index row_end   = rows(alpha + 1);
            for (Index j = row_begin; j < row_end; ++j) {
                sum += vals(j) * x(cols(j));
            }
            y(alpha) = sum;
        }
    );
}

// ====================================================================
//  Host-pointer overload (copies data to/from device, for C API / Python)
// ====================================================================
template <typename ExecSpace>
void MatrixFreeHamiltonian<ExecSpace>::apply(
    const Complex* x,
    Complex* y
) const
{
    const Index dim = dim_;

    // Allocate temporary device views
    VectorView<ExecSpace> x_dev("x_temp", dim);
    VectorView<ExecSpace> y_dev("y_temp", dim);

    // Copy input host → device
    auto x_host = Kokkos::View<const KComplex*,
                               Kokkos::HostSpace,
                               Kokkos::MemoryUnmanaged>(
        reinterpret_cast<const KComplex*>(x), dim);
    Kokkos::deep_copy(ExecSpace(), x_dev, x_host);

    // Run the device kernel
    apply(x_dev, y_dev);

    // Copy result device → host
    auto y_host = Kokkos::View<KComplex*,
                               Kokkos::HostSpace,
                               Kokkos::MemoryUnmanaged>(
        reinterpret_cast<KComplex*>(y), dim);
    Kokkos::deep_copy(ExecSpace(), y_host, y_dev);
}

// ====================================================================
//  Diagonal (host copy)
// ====================================================================
template <typename ExecSpace>
HostVector MatrixFreeHamiltonian<ExecSpace>::diagonal_host() const
{
    HostVector result;
    copy_device_to_host(diagonal_, result);
    return result;
}

// Explicit instantiations
#ifdef KOKKOS_ENABLE_SERIAL
template class MatrixFreeHamiltonian<Kokkos::Serial>;
#endif
#ifdef KOKKOS_ENABLE_OPENMP
template class MatrixFreeHamiltonian<Kokkos::OpenMP>;
#endif
#ifdef KOKKOS_ENABLE_THREADS
template class MatrixFreeHamiltonian<Kokkos::Threads>;
#endif
#ifdef KOKKOS_ENABLE_CUDA
template class MatrixFreeHamiltonian<Kokkos::Cuda>;
#endif
#ifdef KOKKOS_ENABLE_HIP
template class MatrixFreeHamiltonian<Kokkos::HIP>;
#endif
#ifdef KOKKOS_ENABLE_SYCL
template class MatrixFreeHamiltonian<Kokkos::Experimental::SYCL>;
#endif



} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov
