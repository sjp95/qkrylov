#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"
#include "qkrylov/linalg/vector_ops.hpp"
#include <stdexcept>
#include <algorithm>
#include <string>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {

template <typename ExecSpace>
MatrixFreeHamiltonian<ExecSpace>::MatrixFreeHamiltonian(
    std::shared_ptr<Basis> basis,
    std::shared_ptr<Site> site,
    const OpSum& ops,
    Device device
)
    : basis_(std::move(basis)),
      site_(std::move(site)),
      ops_(ops),
      device_(std::move(device))
{
    if (!basis_ || !site_) {
        throw std::invalid_argument("MatrixFreeHamiltonian requires both Basis and Site.");
    }
    if (basis_->bits_per_site() != site_->bits_per_site()) {
        throw std::invalid_argument(
            "Basis and Site dimension mismatch (bits_per_site: " + 
            std::to_string(basis_->bits_per_site()) + " vs " + 
            std::to_string(site_->bits_per_site()) + "). " +
            "You are mixing incompatible physics rules."
        );
    }

    detail::ensure_kokkos_initialized();
    dim_ = basis_->size();
    if (dim_ == 0) return;

    std::vector<KComplex> h_diagonal(dim_, KComplex(0.0, 0.0));

    for (Index alpha = 0; alpha < dim_; ++alpha) {
        const StateID initial_state = basis_->state(alpha);

        for (const auto& term : ops_.terms()) {
            StateID state  = initial_state;
            ComplexDouble amp = term.coeff;
            bool    valid  = true;

            for (const auto& factor : term.factors) {
                auto action = site_->apply(factor.op, factor.site, state);
                if (!action.valid) { valid = false; break; }
                state = action.new_state;
                amp  *= action.matrix_element;
            }

            if (valid && state == initial_state) {
                h_diagonal[alpha] += KComplex(static_cast<Real>(amp.real()), static_cast<Real>(amp.imag()));
            }
        }
    }

    using MemSpace = typename ExecSpace::memory_space;
    diagonal_ = VectorView<ExecSpace>("qkrylov::diagonal", dim_);

    
    auto h_dg = Kokkos::View<const KComplex*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(h_diagonal.data(), dim_);
    Kokkos::deep_copy(ExecSpace(), diagonal_, h_dg);

    // --- NEW: VM Compilation Phase ---
    std::vector<Instruction> host_insts;
    for (const auto& term : ops_.terms()) {
        auto compiled = site_->compile(term);
        for (auto& inst : compiled) {
            // Only store OFF-DIAGONAL instructions in the VM to avoid double-counting
            // the diagonal (which is already fully captured in diagonal_ above).
            if (inst.flip_mask != 0) {
                host_insts.push_back(inst);
            }
        }
    }

    // Sort instructions by bit-flip distance (popcount of flip_mask) to group local interaction terms
    std::sort(host_insts.begin(), host_insts.end(), [](const Instruction& a, const Instruction& b) {
        int pop_a = std::popcount(a.flip_mask);
        int pop_b = std::popcount(b.flip_mask);
        if (pop_a != pop_b) return pop_a < pop_b;
        return a.flip_mask < b.flip_mask;
    });

    num_instructions_ = host_insts.size();
    if (num_instructions_ > 0) {
        instructions_ = Kokkos::View<Instruction*, ExecSpace>("qkrylov::instructions", num_instructions_);
        auto host_view = Kokkos::View<const Instruction*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(host_insts.data(), num_instructions_);
        Kokkos::deep_copy(ExecSpace(), instructions_, host_view);
    }
}



// Safe, fast, portable popcount for Kokkos device code
KOKKOS_INLINE_FUNCTION int qkrylov_popcount64(uint64_t x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __popcll(x);
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    x -= (x >> 1) & 0x5555555555555555ULL;
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
    return (x * 0x0101010101010101ULL) >> 56;
#endif
}

template <typename ExecSpace>
void MatrixFreeHamiltonian<ExecSpace>::apply(
    const VectorView<ExecSpace>& x,
    VectorView<ExecSpace>& y
) const
{
    const Index dim = dim_;
    if (dim == 0) return;
    Kokkos::deep_copy(ExecSpace(), y, KComplex(0.0, 0.0));

    auto* basis_ptr = basis_.get();
    auto  diag_view = diagonal_;
    auto  inst_view = instructions_;
    int   num_inst  = num_instructions_;

    
    Kokkos::parallel_for("qkrylov::H_apply",
        Kokkos::RangePolicy<ExecSpace, Index>(0, dim),
        KOKKOS_LAMBDA(const Index beta) {
            const StateID state_beta = basis_ptr->state(beta);
            
            // 1. Diagonal contribution
            KComplex y_val = diag_view(beta) * x(beta);

            // 2. Off-diagonal VM execution (pull-based, no atomics)
            for (int i = 0; i < num_inst; ++i) {
                const auto& inst = inst_view(i);
                
                StateID state_alpha = state_beta ^ inst.flip_mask;
                
                if ((state_alpha & inst.check_mask) != inst.expected_bits) continue;
                if (!basis_ptr->contains(state_alpha)) continue;
                
                Index alpha = basis_ptr->index(state_alpha);
                
                double real_c = inst.coeff.real();
                double imag_c = inst.coeff.imag();
                
                if (inst.sign_mask) {
                    if (qkrylov_popcount64(state_alpha & inst.sign_mask) % 2 != 0) {
                        real_c = -real_c;
                        imag_c = -imag_c;
                    }
                }
                
                KComplex term_val(static_cast<Real>(real_c), static_cast<Real>(imag_c));
                y_val += term_val * x(alpha);
            }
            y(beta) = y_val;
        }
    );
}


template <typename ExecSpace>
void MatrixFreeHamiltonian<ExecSpace>::apply(
    const Complex* x,
    Complex* y
) const
{
    const Index dim = dim_;
    if (dim == 0) return;

    if (scratch_x_.extent(0) != dim) {
        scratch_x_ = VectorView<ExecSpace>("scratch_x", dim);
        scratch_y_ = VectorView<ExecSpace>("scratch_y", dim);
    }

    auto x_host = Kokkos::View<const KComplex*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(reinterpret_cast<const KComplex*>(x), dim);
    Kokkos::deep_copy(ExecSpace(), scratch_x_, x_host);

    apply(scratch_x_, scratch_y_);

    auto y_host = Kokkos::View<KComplex*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(reinterpret_cast<KComplex*>(y), dim);
    Kokkos::deep_copy(ExecSpace(), y_host, scratch_y_);
}

template <typename ExecSpace>
HostVector MatrixFreeHamiltonian<ExecSpace>::diagonal_host() const
{
    HostVector result;
    copy_device_to_host(diagonal_, result);
    return result;
}

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
