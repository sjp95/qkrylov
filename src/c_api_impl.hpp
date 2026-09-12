#pragma once

#include "qkrylov/c_api.h"
#include "c_api_internal.hpp"

#include "qkrylov/symmetry/sector.hpp"
#include "qkrylov/operators/operator_term.hpp"
#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/basis/basis.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"
#include "qkrylov/basis/fermion_basis.hpp"
#include "qkrylov/basis/hubbard_basis.hpp"
#include "qkrylov/basis/tj_basis.hpp"
#include "qkrylov/basis/spin_s_basis.hpp"
#include "qkrylov/sites/site.hpp"
#include "qkrylov/sites/spinhalf_site.hpp"
#include "qkrylov/sites/spin_s_site.hpp"
#include "qkrylov/sites/fermion_site.hpp"
#include "qkrylov/sites/hubbard_site.hpp"
#include "qkrylov/sites/tj_site.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"
#include "qkrylov/core/device.hpp"
#include "qkrylov/solvers/lanczos.hpp"
#include "qkrylov/solvers/davidson.hpp"
#include "qkrylov/solvers/dynamics.hpp"
#include "qkrylov/solvers/ftlm.hpp"
#include "qkrylov/solvers/correction_vector.hpp"
#include "qkrylov/linalg/vector_ops.hpp"

#include <memory>
#include <vector>
#include <complex>
#include <cstring>
#include <algorithm>
#include <exception>

using namespace qkrylov;
using namespace qkrylov::QKRYLOV_PRECISION_NAMESPACE;

#ifdef QKRYLOV_DOUBLE_PRECISION
#define SUFFIX(name) name##_fp64
#define PREC_ID 1
using Scalar = double;
using LanczosResT = qkrylov_lanczos_result_fp64_t;
using FTLMResT = qkrylov_ftlm_result_fp64_t;
using FTLMSweepResT = qkrylov_ftlm_sweep_result_fp64_t;
using CorrVecResT = qkrylov_correction_vector_result_fp64_t;
#else
#define SUFFIX(name) name##_fp32
#define PREC_ID 0
using Scalar = float;
using LanczosResT = qkrylov_lanczos_result_fp32_t;
using FTLMResT = qkrylov_ftlm_result_fp32_t;
using FTLMSweepResT = qkrylov_ftlm_sweep_result_fp32_t;
using CorrVecResT = qkrylov_correction_vector_result_fp32_t;
#endif

using DevVector = VectorView<Kokkos::DefaultExecutionSpace>;

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {

inline std::shared_ptr<Basis> make_basis_from_descriptor(const qkrylov_basis_t& b) {
    Sector sec;
    sec.use_sz = b.sector.use_sz;
    sec.sz2 = b.sector.sz2;
    sec.use_nup = b.sector.use_nup;
    sec.use_ndn = b.sector.use_ndn;
    sec.nup = b.sector.nup;
    sec.ndn = b.sector.ndn;
    sec.use_n = b.sector.use_n;
    sec.n = b.sector.n;
    sec.use_nb = b.sector.use_nb;
    sec.nb = b.sector.nb;

    switch (b.type) {
        case BasisType::SpinHalf:
            return std::make_shared<SpinHalfBasis>(b.num_sites, sec);
        case BasisType::SpinS:
            return std::make_shared<SpinSBasis>(b.num_sites, b.spin_s, sec);
        case BasisType::Fermion:
            return std::make_shared<FermionBasis>(b.num_sites, sec);
        case BasisType::Hubbard:
            return std::make_shared<HubbardBasis>(b.num_sites, sec);
        case BasisType::TJ:
            return std::make_shared<TJBasis>(b.num_sites, sec);
    }
    return nullptr;
}

inline std::shared_ptr<Site> make_site_from_descriptor(const qkrylov_site_t& s) {
    switch (s.type) {
        case SiteType::SpinHalf:
            return std::make_shared<SpinHalfSite>();
        case SiteType::SpinS:
            return std::make_shared<SpinSSite>(s.spin_s);
        case SiteType::Fermion:
            return std::make_shared<FermionSite>();
        case SiteType::Hubbard:
            return std::make_shared<HubbardSite>();
        case SiteType::TJ:
            return std::make_shared<TJSite>();
    }
    return nullptr;
}

inline OpSum make_opsum_from_descriptor(const qkrylov_opsum_t& o) {
    OpSum ops;
    for (const auto& term_desc : o.terms) {
        OperatorTerm term;
        term.coeff = Complex(static_cast<Real>(term_desc.coeff_real),
                             static_cast<Real>(term_desc.coeff_imag));
        for (const auto& f : term_desc.factors) {
            term.factors.push_back({f.first, f.second});
        }
        ops.add_term(term);
    }
    return ops;
}

#ifdef QKRYLOV_DOUBLE_PRECISION
inline std::shared_ptr<Basis> get_or_create_basis(const qkrylov_basis_t& b) {
    if (b.ptr64) {
        return std::static_pointer_cast<Basis>(b.ptr64);
    }
    if (b.ptr32) {
        auto ptr = std::static_pointer_cast<Basis>(b.ptr32);
        b.ptr64 = ptr;
        return ptr;
    }
    auto ptr = make_basis_from_descriptor(b);
    b.ptr64 = ptr;
    b.ptr32 = ptr;
    if (ptr) b.cached_dim = ptr->size();
    return ptr;
}
inline std::shared_ptr<Site> get_or_create_site(const qkrylov_site_t& s) {
    if (s.ptr64) {
        return std::static_pointer_cast<Site>(s.ptr64);
    }
    auto ptr = make_site_from_descriptor(s);
    s.ptr64 = ptr;
    return ptr;
}
#else
inline std::shared_ptr<Basis> get_or_create_basis(const qkrylov_basis_t& b) {
    if (b.ptr32) {
        return std::static_pointer_cast<Basis>(b.ptr32);
    }
    if (b.ptr64) {
        auto ptr = std::static_pointer_cast<Basis>(b.ptr64);
        b.ptr32 = ptr;
        return ptr;
    }
    auto ptr = make_basis_from_descriptor(b);
    b.ptr32 = ptr;
    b.ptr64 = ptr;
    if (ptr) b.cached_dim = ptr->size();
    return ptr;
}
inline std::shared_ptr<Site> get_or_create_site(const qkrylov_site_t& s) {
    if (s.ptr32) {
        return std::static_pointer_cast<Site>(s.ptr32);
    }
    auto ptr = make_site_from_descriptor(s);
    s.ptr32 = ptr;
    return ptr;
}
#endif

} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov

namespace {

template <typename Func>
inline auto run_unary_read(uint64_t dim, const Scalar* x, Func&& op) {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    auto x_host = Kokkos::View<const KComplex*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
        reinterpret_cast<const KComplex*>(x), dim);
    if constexpr (Kokkos::SpaceAccessibility<ExecSpace, Kokkos::HostSpace>::accessible) {
        return op(x_host);
    } else {
        VectorView<ExecSpace> x_dev("x_dev", dim);
        Kokkos::deep_copy(ExecSpace(), x_dev, x_host);
        return op(x_dev);
    }
}

template <typename Func>
inline void run_unary_mut(uint64_t dim, Scalar* x, Func&& op) {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    auto x_host = Kokkos::View<KComplex*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
        reinterpret_cast<KComplex*>(x), dim);
    if constexpr (Kokkos::SpaceAccessibility<ExecSpace, Kokkos::HostSpace>::accessible) {
        op(x_host);
    } else {
        VectorView<ExecSpace> x_dev("x_dev", dim);
        Kokkos::deep_copy(ExecSpace(), x_dev, x_host);
        op(x_dev);
        Kokkos::deep_copy(ExecSpace(), x_host, x_dev);
    }
}

template <typename Func>
inline auto run_binary_read(uint64_t dim, const Scalar* x, const Scalar* y, Func&& op) {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    auto x_host = Kokkos::View<const KComplex*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
        reinterpret_cast<const KComplex*>(x), dim);
    auto y_host = Kokkos::View<const KComplex*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
        reinterpret_cast<const KComplex*>(y), dim);
    if constexpr (Kokkos::SpaceAccessibility<ExecSpace, Kokkos::HostSpace>::accessible) {
        return op(x_host, y_host);
    } else {
        VectorView<ExecSpace> x_dev("x_dev", dim);
        VectorView<ExecSpace> y_dev("y_dev", dim);
        Kokkos::deep_copy(ExecSpace(), x_dev, x_host);
        Kokkos::deep_copy(ExecSpace(), y_dev, y_host);
        return op(x_dev, y_dev);
    }
}

template <typename Func>
inline void run_binary_mut(uint64_t dim, const Scalar* x, Scalar* y, Func&& op) {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    auto x_host = Kokkos::View<KComplex*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
        const_cast<KComplex*>(reinterpret_cast<const KComplex*>(x)), dim);
    auto y_host = Kokkos::View<KComplex*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
        reinterpret_cast<KComplex*>(y), dim);
    if constexpr (Kokkos::SpaceAccessibility<ExecSpace, Kokkos::HostSpace>::accessible) {
        op(x_host, y_host);
    } else {
        VectorView<ExecSpace> x_dev("x_dev", dim);
        VectorView<ExecSpace> y_dev("y_dev", dim);
        Kokkos::deep_copy(ExecSpace(), x_dev, x_host);
        Kokkos::deep_copy(ExecSpace(), y_dev, y_host);
        op(x_dev, y_dev);
        Kokkos::deep_copy(ExecSpace(), y_host, y_dev);
    }
}

} // anonymous namespace

extern "C" {

qkrylov_hamiltonian_h SUFFIX(qkrylov_hamiltonian_create)(
    qkrylov_basis_h basis,
    qkrylov_site_h site,
    qkrylov_opsum_h opsum)
{
    return SUFFIX(qkrylov_hamiltonian_create_device)(basis, site, opsum, "cpu");
}

qkrylov_hamiltonian_h SUFFIX(qkrylov_hamiltonian_create_device)(
    qkrylov_basis_h basis,
    qkrylov_site_h site,
    qkrylov_opsum_h opsum,
    const char* device_str)
{
    if (!basis || !site || !opsum) {
        set_last_error("qkrylov_hamiltonian_create_device: null basis, site, or opsum handle");
        return nullptr;
    }
    try {
        std::string dev_str = device_str ? device_str : "cpu";
        auto b_ptr = get_or_create_basis(*basis);
        auto s_ptr = get_or_create_site(*site);
        auto ops = make_opsum_from_descriptor(*opsum);

        if (!b_ptr || !s_ptr) {
            set_last_error("qkrylov_hamiltonian_create_device: failed to create basis or site");
            return nullptr;
        }

        auto H = std::make_shared<MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace>>(
            b_ptr, s_ptr, ops, Device(dev_str)
        );

        auto handle = std::make_unique<qkrylov_hamiltonian_t>();
        handle->precision = PREC_ID;
        handle->dim = H->dimension();
        handle->impl = H;
        return handle.release();
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_hamiltonian_create_device");
        return nullptr;
    }
}

int SUFFIX(qkrylov_hamiltonian_apply)(
    qkrylov_hamiltonian_h h,
    const Scalar* x_real, const Scalar* x_imag,
    Scalar* y_real, Scalar* y_imag)
{
    if (!h) {
        set_last_error("qkrylov_hamiltonian_apply: hamiltonian handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (h->precision != PREC_ID) {
        set_last_error("qkrylov_hamiltonian_apply: precision mismatch");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (!h->impl || !x_real || !y_real) {
        set_last_error("qkrylov_hamiltonian_apply: null vector pointer or uninitialized hamiltonian");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        auto* H = static_cast<MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace>*>(h->impl.get());
        const uint64_t dim = h->dim;
        std::vector<Complex> x(dim);
        std::vector<Complex> y(dim);

        for (uint64_t i = 0; i < dim; ++i) {
            Scalar imag = x_imag ? x_imag[i] : Scalar(0.0);
            x[i] = Complex(static_cast<Real>(x_real[i]), static_cast<Real>(imag));
        }

        H->apply(x.data(), y.data());

        for (uint64_t i = 0; i < dim; ++i) {
            y_real[i] = static_cast<Scalar>(y[i].real());
            if (y_imag) y_imag[i] = static_cast<Scalar>(y[i].imag());
        }

        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_hamiltonian_apply");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int SUFFIX(qkrylov_hamiltonian_apply_complex)(
    qkrylov_hamiltonian_h h,
    const Scalar* x_complex,
    Scalar* y_complex)
{
    if (!h) {
        set_last_error("qkrylov_hamiltonian_apply_complex: hamiltonian handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (h->precision != PREC_ID) {
        set_last_error("qkrylov_hamiltonian_apply_complex: precision mismatch");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (!h->impl || !x_complex || !y_complex) {
        set_last_error("qkrylov_hamiltonian_apply_complex: null vector pointer or uninitialized hamiltonian");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        auto* H = static_cast<MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace>*>(h->impl.get());
        const auto* x_c = reinterpret_cast<const Complex*>(x_complex);
        auto* y_c = reinterpret_cast<Complex*>(y_complex);
        H->apply(x_c, y_c);
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_hamiltonian_apply_complex");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int SUFFIX(qkrylov_hamiltonian_diagonal)(
    qkrylov_hamiltonian_h h,
    Scalar* diag_out)
{
    if (!h) {
        set_last_error("qkrylov_hamiltonian_diagonal: hamiltonian handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (h->precision != PREC_ID) {
        set_last_error("qkrylov_hamiltonian_diagonal: precision mismatch");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (!h->impl || !diag_out) {
        set_last_error("qkrylov_hamiltonian_diagonal: null pointer or uninitialized hamiltonian");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        auto* H = static_cast<MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace>*>(h->impl.get());
        auto diag = H->diagonal_host();
        const uint64_t dim = h->dim;
        for (uint64_t i = 0; i < dim && i < diag.size(); ++i) {
            diag_out[i] = static_cast<Scalar>(diag[i].real());
        }
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_hamiltonian_diagonal");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int SUFFIX(qkrylov_hamiltonian_apply_device)(
    qkrylov_hamiltonian_h h,
    const qkrylov_device_vector_h x_dev,
    qkrylov_device_vector_h y_dev)
{
    if (!h) {
        set_last_error("qkrylov_hamiltonian_apply_device: hamiltonian handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (h->precision != PREC_ID) {
        set_last_error("qkrylov_hamiltonian_apply_device: hamiltonian precision mismatch");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (!x_dev || !y_dev) {
        set_last_error("qkrylov_hamiltonian_apply_device: device vector handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (x_dev->precision != PREC_ID || y_dev->precision != PREC_ID) {
        set_last_error("qkrylov_hamiltonian_apply_device: device vector precision mismatch");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (x_dev->dim != h->dim || y_dev->dim != h->dim) {
        set_last_error("qkrylov_hamiltonian_apply_device: vector dimension does not match hamiltonian dimension");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (!h->impl || !x_dev->impl || !y_dev->impl) {
        set_last_error("qkrylov_hamiltonian_apply_device: uninitialized handle implementation");
        return QKRYLOV_ERROR_INVALID_ARG;
    }

    try {
        auto* H = static_cast<MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace>*>(h->impl.get());
        const auto* x_v = static_cast<const DevVector*>(x_dev->impl.get());
        auto* y_v = static_cast<DevVector*>(y_dev->impl.get());

        H->apply(*x_v, *y_v);
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_hamiltonian_apply_device");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int SUFFIX(qkrylov_hamiltonian_diagonal_device)(
    qkrylov_hamiltonian_h h,
    qkrylov_device_vector_h diag_out)
{
    if (!h) {
        set_last_error("qkrylov_hamiltonian_diagonal_device: hamiltonian handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (h->precision != PREC_ID) {
        set_last_error("qkrylov_hamiltonian_diagonal_device: hamiltonian precision mismatch");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (!diag_out) {
        set_last_error("qkrylov_hamiltonian_diagonal_device: diagonal vector handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (diag_out->precision != PREC_ID) {
        set_last_error("qkrylov_hamiltonian_diagonal_device: diagonal vector precision mismatch");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (diag_out->dim != h->dim) {
        set_last_error("qkrylov_hamiltonian_diagonal_device: vector dimension does not match hamiltonian dimension");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (!h->impl || !diag_out->impl) {
        set_last_error("qkrylov_hamiltonian_diagonal_device: uninitialized handle implementation");
        return QKRYLOV_ERROR_INVALID_ARG;
    }

    try {
        auto* H = static_cast<MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace>*>(h->impl.get());
        auto* d_v = static_cast<DevVector*>(diag_out->impl.get());

        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), *d_v, H->diagonal());
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_hamiltonian_diagonal_device");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int SUFFIX(qkrylov_lanczos_ground_state)(
    qkrylov_hamiltonian_h h,
    int maxiter,
    Scalar tol,
    LanczosResT* result)
{
    return SUFFIX(qkrylov_lanczos_ground_state_complex)(h, maxiter, tol, result, nullptr);
}

int SUFFIX(qkrylov_lanczos_ground_state_complex)(
    qkrylov_hamiltonian_h h,
    int maxiter,
    Scalar tol,
    LanczosResT* result,
    Scalar* eigenvector_complex)
{
    if (!h) {
        set_last_error("qkrylov_lanczos_ground_state_complex: hamiltonian handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (h->precision != PREC_ID) {
        set_last_error("qkrylov_lanczos_ground_state_complex: precision mismatch");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (!h->impl || !result) {
        set_last_error("qkrylov_lanczos_ground_state_complex: null result pointer or uninitialized hamiltonian");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (maxiter <= 0) {
        set_last_error("qkrylov_lanczos_ground_state_complex: maxiter must be positive");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        auto* H = static_cast<MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace>*>(h->impl.get());
        LanczosConfig cfg;
        cfg.maxiter = maxiter;
        cfg.tol = static_cast<Real>(tol);
        auto res = eigenvector_complex
            ? solvers::lanczos<solvers::policy::OnePass_DKGS>(*H, cfg)
            : solvers::lanczos<solvers::policy::OnePass>(*H, cfg);
        result->energy     = static_cast<Scalar>(res.energy);
        result->iterations = res.iterations;
        result->converged  = res.converged ? 1 : 0;
        if (eigenvector_complex && !res.eigenvector.empty()) {
            for (size_t i = 0; i < res.eigenvector.size(); ++i) {
                eigenvector_complex[2 * i]     = static_cast<Scalar>(res.eigenvector[i].real());
                eigenvector_complex[2 * i + 1] = static_cast<Scalar>(res.eigenvector[i].imag());
            }
        }
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_lanczos_ground_state_complex");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int SUFFIX(qkrylov_lanczos_two_pass_ground_state)(
    qkrylov_hamiltonian_h h,
    int maxiter,
    Scalar tol,
    LanczosResT* result)
{
    return SUFFIX(qkrylov_lanczos_two_pass_ground_state_complex)(h, maxiter, tol, result, nullptr);
}

int SUFFIX(qkrylov_lanczos_two_pass_ground_state_complex)(
    qkrylov_hamiltonian_h h,
    int maxiter,
    Scalar tol,
    LanczosResT* result,
    Scalar* eigenvector_complex)
{
    if (!h) {
        set_last_error("qkrylov_lanczos_two_pass_ground_state_complex: hamiltonian handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (h->precision != PREC_ID) {
        set_last_error("qkrylov_lanczos_two_pass_ground_state_complex: precision mismatch");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (!h->impl || !result) {
        set_last_error("qkrylov_lanczos_two_pass_ground_state_complex: null result pointer or uninitialized hamiltonian");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (maxiter <= 0) {
        set_last_error("qkrylov_lanczos_two_pass_ground_state_complex: maxiter must be positive");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        auto* H = static_cast<MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace>*>(h->impl.get());
        LanczosConfig cfg;
        cfg.maxiter = maxiter;
        cfg.tol = static_cast<Real>(tol);
        auto res = solvers::lanczos<solvers::policy::TwoPass>(*H, cfg);
        result->energy     = static_cast<Scalar>(res.energy);
        result->iterations = res.iterations;
        result->converged  = res.converged ? 1 : 0;
        if (eigenvector_complex && !res.eigenvector.empty()) {
            for (size_t i = 0; i < res.eigenvector.size(); ++i) {
                eigenvector_complex[2 * i]     = static_cast<Scalar>(res.eigenvector[i].real());
                eigenvector_complex[2 * i + 1] = static_cast<Scalar>(res.eigenvector[i].imag());
            }
        }
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_lanczos_two_pass_ground_state_complex");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int SUFFIX(qkrylov_lanczos_lowest_complex)(
    qkrylov_hamiltonian_h h,
    int n_eig,
    int maxiter,
    Scalar tol,
    Scalar* eigenvalues_out,
    Scalar* eigenvectors_complex_out,
    qkrylov_lanczos_lowest_result_c_t* result_info,
    const Scalar* initial_vector_complex)
{
    if (!h) {
        set_last_error("qkrylov_lanczos_lowest_complex: hamiltonian handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (h->precision != PREC_ID) {
        set_last_error("qkrylov_lanczos_lowest_complex: precision mismatch");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (!h->impl || !eigenvalues_out) {
        set_last_error("qkrylov_lanczos_lowest_complex: null eigenvalues output pointer or uninitialized hamiltonian");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (n_eig <= 0) {
        set_last_error("qkrylov_lanczos_lowest_complex: n_eig must be positive");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (maxiter <= 0) {
        set_last_error("qkrylov_lanczos_lowest_complex: maxiter must be positive");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        auto* H = static_cast<MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace>*>(h->impl.get());
        bool compute_evecs = (eigenvectors_complex_out != nullptr);
        HostVector init_v;
        if (initial_vector_complex) {
            init_v.resize(h->dim);
            for (uint64_t i = 0; i < h->dim; ++i) {
                init_v[i] = Complex(static_cast<Real>(initial_vector_complex[2 * i]),
                                    static_cast<Real>(initial_vector_complex[2 * i + 1]));
            }
        }
        LanczosConfig cfg;
        cfg.n_eig = n_eig;
        cfg.maxiter = maxiter;
        cfg.tol = static_cast<Real>(tol);
        cfg.compute_eigenvectors = compute_evecs;
        cfg.initial_vector = std::move(init_v);
        auto res = solvers::lanczos_lowest(*H, cfg);
        const size_t k = std::min(static_cast<size_t>(n_eig), res.eigenvalues.size());
        for (size_t i = 0; i < k; ++i) {
            eigenvalues_out[i] = static_cast<Scalar>(res.eigenvalues[i]);
        }

        if (result_info) {
            result_info->iterations = res.iterations;
            result_info->converged  = res.converged ? 1 : 0;
        }

        if (compute_evecs) {
            const uint64_t dim = h->dim;
            for (size_t idx = 0; idx < k && idx < res.eigenvectors.size(); ++idx) {
                const auto& vec = res.eigenvectors[idx];
                Scalar* dst = eigenvectors_complex_out + (idx * 2 * dim);
                for (size_t i = 0; i < dim && i < vec.size(); ++i) {
                    dst[2 * i]     = static_cast<Scalar>(vec[i].real());
                    dst[2 * i + 1] = static_cast<Scalar>(vec[i].imag());
                }
            }
        }
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_lanczos_lowest_complex");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int SUFFIX(qkrylov_davidson_lowest_complex)(
    qkrylov_hamiltonian_h h,
    int n_eig,
    int max_subspace,
    Scalar tol,
    Scalar* eigenvalues_out,
    Scalar* eigenvectors_complex_out,
    qkrylov_davidson_result_c_t* result_info)
{
    if (!h) {
        set_last_error("qkrylov_davidson_lowest_complex: hamiltonian handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (h->precision != PREC_ID) {
        set_last_error("qkrylov_davidson_lowest_complex: precision mismatch");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (!h->impl || !eigenvalues_out) {
        set_last_error("qkrylov_davidson_lowest_complex: null eigenvalues output pointer or uninitialized hamiltonian");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (n_eig <= 0) {
        set_last_error("qkrylov_davidson_lowest_complex: n_eig must be positive");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (max_subspace <= 0) {
        set_last_error("qkrylov_davidson_lowest_complex: max_subspace must be positive");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        auto* H = static_cast<MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace>*>(h->impl.get());
        auto res = davidson_lowest(*H, n_eig, max_subspace, static_cast<Real>(tol));
        const size_t k = std::min(static_cast<size_t>(n_eig), res.eigenvalues.size());
        for (size_t i = 0; i < k; ++i) {
            eigenvalues_out[i] = static_cast<Scalar>(res.eigenvalues[i]);
        }

        if (result_info) {
            result_info->iterations = res.iterations;
            result_info->converged  = res.converged ? 1 : 0;
        }

        if (eigenvectors_complex_out) {
            const uint64_t dim = h->dim;
            for (size_t idx = 0; idx < k && idx < res.eigenvectors.size(); ++idx) {
                const auto& vec = res.eigenvectors[idx];
                Scalar* dst = eigenvectors_complex_out + (idx * 2 * dim);
                for (size_t i = 0; i < dim && i < vec.size(); ++i) {
                    dst[2 * i]     = static_cast<Scalar>(vec[i].real());
                    dst[2 * i + 1] = static_cast<Scalar>(vec[i].imag());
                }
            }
        }
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_davidson_lowest_complex");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int SUFFIX(qkrylov_continued_fraction_coeffs_complex)(
    qkrylov_hamiltonian_h h,
    const Scalar* phi0_complex,
    int n_iter,
    Scalar* alphas_out,
    Scalar* betas_out,
    Scalar* norm_phi0_out,
    int* num_coeffs_out)
{
    if (!h) {
        set_last_error("qkrylov_continued_fraction_coeffs_complex: hamiltonian handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (h->precision != PREC_ID) {
        set_last_error("qkrylov_continued_fraction_coeffs_complex: precision mismatch");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (!h->impl || !phi0_complex || !alphas_out || !betas_out || !norm_phi0_out) {
        set_last_error("qkrylov_continued_fraction_coeffs_complex: null pointer argument");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (n_iter <= 0) {
        set_last_error("qkrylov_continued_fraction_coeffs_complex: n_iter must be positive");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        auto* H = static_cast<MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace>*>(h->impl.get());
        const uint64_t dim = h->dim;
        HostVector phi0(dim);
        for (uint64_t i = 0; i < dim; ++i) {
            phi0[i] = Complex(static_cast<Real>(phi0_complex[2 * i]), static_cast<Real>(phi0_complex[2 * i + 1]));
        }

        auto res = continued_fraction_coeffs(*H, phi0, n_iter);
        *norm_phi0_out = static_cast<Scalar>(res.norm_phi0);

        const size_t n_alpha = res.alphas.size();
        const size_t n_beta  = res.betas.size();

        for (size_t i = 0; i < n_alpha && i < static_cast<size_t>(n_iter); ++i) {
            alphas_out[i] = static_cast<Scalar>(res.alphas[i]);
        }
        for (size_t i = 0; i < n_beta && i < static_cast<size_t>(n_iter); ++i) {
            betas_out[i] = static_cast<Scalar>(res.betas[i]);
        }

        if (num_coeffs_out) {
            *num_coeffs_out = static_cast<int>(n_alpha);
        }

        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_continued_fraction_coeffs_complex");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

Scalar SUFFIX(qkrylov_evaluate_spectral_function)(
    const Scalar* alphas,
    const Scalar* betas,
    size_t n,
    Scalar norm_phi0,
    Scalar omega,
    Scalar E0,
    Scalar eta)
{
    if (!alphas || !betas || n == 0) {
        set_last_error("qkrylov_evaluate_spectral_function: invalid arguments (null pointer or n == 0)");
        return Scalar(0.0);
    }
    try {
        std::vector<Real> alphas_real(n);
        std::vector<Real> betas_real(n > 0 ? n - 1 : 0);
        for (size_t i = 0; i < n; ++i) {
            alphas_real[i] = static_cast<Real>(alphas[i]);
        }
        size_t n_beta = n > 0 ? n - 1 : 0;
        for (size_t i = 0; i < n_beta; ++i) {
            betas_real[i] = static_cast<Real>(betas[i]);
        }
        Real val = evaluate_spectral_function(alphas_real.data(), betas_real.data(), n,
                                              static_cast<Real>(norm_phi0),
                                              static_cast<Real>(omega),
                                              static_cast<Real>(E0),
                                              static_cast<Real>(eta));
        return static_cast<Scalar>(val);
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return Scalar(0.0);
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_evaluate_spectral_function");
        return Scalar(0.0);
    }
}

int SUFFIX(qkrylov_ftlm)(
    qkrylov_hamiltonian_h h,
    Scalar beta,
    int n_random,
    int n_steps,
    FTLMResT* result)
{
    if (!h) {
        set_last_error("qkrylov_ftlm: hamiltonian handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (h->precision != PREC_ID) {
        set_last_error("qkrylov_ftlm: precision mismatch");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (!h->impl || !result) {
        set_last_error("qkrylov_ftlm: null pointer argument");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (n_random <= 0 || n_steps <= 0) {
        set_last_error("qkrylov_ftlm: n_random and n_steps must be positive");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        auto* H = static_cast<MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace>*>(h->impl.get());
        auto res = ftlm(*H, static_cast<Real>(beta), n_random, n_steps);
        result->beta               = static_cast<Scalar>(res.beta);
        result->partition_function = static_cast<Scalar>(res.partition_function);
        result->internal_energy    = static_cast<Scalar>(res.internal_energy);
        result->specific_heat      = static_cast<Scalar>(res.specific_heat);
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_ftlm");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int SUFFIX(qkrylov_ftlm_sweep)(
    qkrylov_hamiltonian_h h,
    const Scalar* beta_grid,
    int num_betas,
    const qkrylov_hamiltonian_h* observables,
    int num_observables,
    int n_random,
    int n_steps,
    uint64_t seed,
    FTLMSweepResT* result)
{
    if (!h) {
        set_last_error("qkrylov_ftlm_sweep: hamiltonian handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (h->precision != PREC_ID) {
        set_last_error("qkrylov_ftlm_sweep: precision mismatch");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (!h->impl || !beta_grid || num_betas <= 0 || !result) {
        set_last_error("qkrylov_ftlm_sweep: null pointer or invalid beta_grid");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (n_random <= 0 || n_steps <= 0) {
        set_last_error("qkrylov_ftlm_sweep: n_random and n_steps must be positive");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (num_observables > 0 && !observables) {
        set_last_error("qkrylov_ftlm_sweep: observables array is null but num_observables > 0");
        return QKRYLOV_ERROR_INVALID_ARG;
    }

    try {
        auto* H = static_cast<MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace>*>(h->impl.get());
        std::vector<Real> betas(num_betas);
        for (int i = 0; i < num_betas; ++i) betas[i] = static_cast<Real>(beta_grid[i]);

        std::vector<MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace>> obs_vec;
        obs_vec.reserve(num_observables);
        for (int i = 0; i < num_observables; ++i) {
            if (!observables[i] || observables[i]->precision != PREC_ID || !observables[i]->impl) {
                set_last_error("qkrylov_ftlm_sweep: invalid observable handle");
                return QKRYLOV_ERROR_INVALID_ARG;
            }
            obs_vec.push_back(*static_cast<MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace>*>(observables[i]->impl.get()));
        }

        auto sweep = ftlm_sweep<Kokkos::DefaultExecutionSpace>(*H, betas, obs_vec, n_random, n_steps, seed);

        result->num_betas = num_betas;
        result->num_observables = num_observables;

        auto* out_betas = new Scalar[num_betas];
        auto* out_z = new Scalar[num_betas];
        auto* out_f = new Scalar[num_betas];
        auto* out_e = new Scalar[num_betas];
        auto* out_cv = new Scalar[num_betas];
        auto* out_s = new Scalar[num_betas];
        for (int bi = 0; bi < num_betas; ++bi) {
            out_betas[bi] = static_cast<Scalar>(sweep.beta_grid[bi]);
            out_z[bi] = static_cast<Scalar>(sweep.partition_functions[bi]);
            out_f[bi] = static_cast<Scalar>(sweep.free_energies[bi]);
            out_e[bi] = static_cast<Scalar>(sweep.internal_energies[bi]);
            out_cv[bi] = static_cast<Scalar>(sweep.specific_heats[bi]);
            out_s[bi] = static_cast<Scalar>(sweep.entropies[bi]);
        }
        result->beta_grid = out_betas;
        result->partition_functions = out_z;
        result->free_energies = out_f;
        result->internal_energies = out_e;
        result->specific_heats = out_cv;
        result->entropies = out_s;

        if (num_observables > 0) {
            auto* out_obs = new Scalar[num_observables * num_betas];
            auto* out_err = new Scalar[num_observables * num_betas];
            for (int oi = 0; oi < num_observables; ++oi) {
                for (int bi = 0; bi < num_betas; ++bi) {
                    out_obs[oi * num_betas + bi] = static_cast<Scalar>(sweep.observable_expectations[oi][bi]);
                    out_err[oi * num_betas + bi] = static_cast<Scalar>(sweep.observable_errors[oi][bi]);
                }
            }
            result->observable_expectations = out_obs;
            result->observable_errors = out_err;
        } else {
            result->observable_expectations = nullptr;
            result->observable_errors = nullptr;
        }

        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_ftlm_sweep");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

void SUFFIX(qkrylov_ftlm_sweep_result_free)(FTLMSweepResT* result) {
    if (!result) return;
    delete[] result->beta_grid;
    delete[] result->partition_functions;
    delete[] result->free_energies;
    delete[] result->internal_energies;
    delete[] result->specific_heats;
    delete[] result->entropies;
    delete[] result->observable_expectations;
    delete[] result->observable_errors;
    result->beta_grid = nullptr;
    result->partition_functions = nullptr;
    result->free_energies = nullptr;
    result->internal_energies = nullptr;
    result->specific_heats = nullptr;
    result->entropies = nullptr;
    result->observable_expectations = nullptr;
    result->observable_errors = nullptr;
    result->num_betas = 0;
    result->num_observables = 0;
}

int SUFFIX(qkrylov_solver_correction_vector)(
    qkrylov_hamiltonian_h h,
    const Scalar* op_psi0_complex,
    Scalar e0,
    Scalar omega,
    Scalar eta,
    int max_iter,
    Scalar tol,
    CorrVecResT* result,
    Scalar* correction_vector_out_complex)
{
    if (!h) {
        set_last_error("qkrylov_solver_correction_vector: hamiltonian handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (h->precision != PREC_ID) {
        set_last_error("qkrylov_solver_correction_vector: precision mismatch");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (!h->impl || !op_psi0_complex || !result) {
        set_last_error("qkrylov_solver_correction_vector: null pointer argument");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (max_iter <= 0) {
        set_last_error("qkrylov_solver_correction_vector: max_iter must be positive");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        auto* H = static_cast<MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace>*>(h->impl.get());
        const uint64_t dim = h->dim;
        HostVector op_psi0(dim);
        for (uint64_t i = 0; i < dim; ++i) {
            op_psi0[i] = Complex(static_cast<Real>(op_psi0_complex[2 * i]),
                                 static_cast<Real>(op_psi0_complex[2 * i + 1]));
        }

        auto cv_res = correction_vector_spectral(
            *H,
            op_psi0,
            static_cast<Real>(e0),
            static_cast<Real>(omega),
            static_cast<Real>(eta),
            max_iter,
            static_cast<Real>(tol)
        );

        result->spectral_function = static_cast<Scalar>(cv_res.spectral_function);
        result->iterations        = cv_res.iterations;
        result->converged         = cv_res.converged ? 1 : 0;

        if (correction_vector_out_complex) {
            for (uint64_t i = 0; i < dim && i < cv_res.correction_vector.size(); ++i) {
                correction_vector_out_complex[2 * i]     = static_cast<Scalar>(cv_res.correction_vector[i].real());
                correction_vector_out_complex[2 * i + 1] = static_cast<Scalar>(cv_res.correction_vector[i].imag());
            }
        }

        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_solver_correction_vector");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

/* ========================================================================= */
/* Vector Operations (Kokkos Parallel BLAS-1 Kernels)                        */
/* ========================================================================= */

int SUFFIX(qkrylov_vector_dot)(
    uint64_t dim,
    const Scalar* x_complex,
    const Scalar* y_complex,
    Scalar* dot_re,
    Scalar* dot_im)
{
    if (!dot_re || !dot_im) {
        set_last_error("qkrylov_vector_dot: output pointer is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (dim == 0) {
        *dot_re = 0;
        *dot_im = 0;
        return QKRYLOV_SUCCESS;
    }
    if (!x_complex || !y_complex) {
        set_last_error("qkrylov_vector_dot: null vector pointer");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        detail::initialize_kokkos();
        KComplex res = run_binary_read(dim, x_complex, y_complex, [](const auto& xv, const auto& yv) {
            return dot(xv, yv);
        });
        *dot_re = static_cast<Scalar>(res.real());
        *dot_im = static_cast<Scalar>(res.imag());
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_vector_dot");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int SUFFIX(qkrylov_vector_norm)(
    uint64_t dim,
    const Scalar* x_complex,
    Scalar* norm_out)
{
    if (!norm_out) {
        set_last_error("qkrylov_vector_norm: output pointer is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (dim == 0) {
        *norm_out = 0;
        return QKRYLOV_SUCCESS;
    }
    if (!x_complex) {
        set_last_error("qkrylov_vector_norm: null vector pointer");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        detail::initialize_kokkos();
        Real res = run_unary_read(dim, x_complex, [](const auto& xv) {
            return norm(xv);
        });
        *norm_out = static_cast<Scalar>(res);
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_vector_norm");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int SUFFIX(qkrylov_vector_axpy)(
    uint64_t dim,
    Scalar a_re,
    Scalar a_im,
    const Scalar* x_complex,
    Scalar* y_complex)
{
    if (dim == 0) {
        return QKRYLOV_SUCCESS;
    }
    if (!x_complex || !y_complex) {
        set_last_error("qkrylov_vector_axpy: null vector pointer");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        detail::initialize_kokkos();
        KComplex a(static_cast<Real>(a_re), static_cast<Real>(a_im));
        run_binary_mut(dim, x_complex, y_complex, [a](const auto& xv, auto& yv) {
            axpy(a, xv, yv);
        });
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_vector_axpy");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int SUFFIX(qkrylov_vector_scal)(
    uint64_t dim,
    Scalar a_re,
    Scalar a_im,
    Scalar* x_complex)
{
    if (dim == 0) {
        return QKRYLOV_SUCCESS;
    }
    if (!x_complex) {
        set_last_error("qkrylov_vector_scal: null vector pointer");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        detail::initialize_kokkos();
        KComplex a(static_cast<Real>(a_re), static_cast<Real>(a_im));
        run_unary_mut(dim, x_complex, [a](auto& xv) {
            scal(a, xv);
        });
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_vector_scal");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int SUFFIX(qkrylov_vector_normalize)(
    uint64_t dim,
    Scalar* x_complex)
{
    if (dim == 0) {
        return QKRYLOV_SUCCESS;
    }
    if (!x_complex) {
        set_last_error("qkrylov_vector_normalize: null vector pointer");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        detail::initialize_kokkos();
        run_unary_mut(dim, x_complex, [](auto& xv) {
            normalize(xv);
        });
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_vector_normalize");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int SUFFIX(qkrylov_vector_zero_fill)(
    uint64_t dim,
    Scalar* x_complex)
{
    if (dim == 0) {
        return QKRYLOV_SUCCESS;
    }
    if (!x_complex) {
        set_last_error("qkrylov_vector_zero_fill: null vector pointer");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        detail::initialize_kokkos();
        run_unary_mut(dim, x_complex, [](auto& xv) {
            zero_fill(xv);
        });
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_vector_zero_fill");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int SUFFIX(qkrylov_vector_copy)(
    uint64_t dim,
    const Scalar* src_complex,
    Scalar* dst_complex)
{
    if (dim == 0) {
        return QKRYLOV_SUCCESS;
    }
    if (!src_complex || !dst_complex) {
        set_last_error("qkrylov_vector_copy: null vector pointer");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        detail::initialize_kokkos();
        run_binary_mut(dim, src_complex, dst_complex, [](const auto& sv, auto& dv) {
            deep_copy(dv, sv);
        });
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_vector_copy");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

/* -----------------------------------------------------------------------------
 * Device-Resident Vector Operations
 * ----------------------------------------------------------------------------- */

qkrylov_device_vector_h SUFFIX(qkrylov_device_vector_create)(uint64_t dim) {
    try {
        detail::initialize_kokkos();
        auto dev_vec = std::make_shared<DevVector>("qkrylov_device_vector", dim);
        auto* handle = new qkrylov_device_vector_t();
        handle->precision = PREC_ID;
        handle->dim = dim;
        handle->impl = dev_vec;
        return handle;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_device_vector_create");
        return nullptr;
    }
}

int SUFFIX(qkrylov_device_vector_copy_from_host)(
    qkrylov_device_vector_h dst,
    const Scalar* host_src_complex)
{
    if (!dst) {
        set_last_error("qkrylov_device_vector_copy_from_host: vector handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (dst->precision != PREC_ID) {
        set_last_error("qkrylov_device_vector_copy_from_host: precision mismatch");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (!dst->impl) {
        set_last_error("qkrylov_device_vector_copy_from_host: uninitialized device vector");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (dst->dim > 0 && !host_src_complex) {
        set_last_error("qkrylov_device_vector_copy_from_host: host pointer is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (dst->dim == 0) return QKRYLOV_SUCCESS;

    try {
        auto* d_vec = static_cast<DevVector*>(dst->impl.get());
        auto h_view = Kokkos::View<const KComplex*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
            reinterpret_cast<const KComplex*>(host_src_complex),
            dst->dim
        );
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), *d_vec, h_view);
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_device_vector_copy_from_host");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int SUFFIX(qkrylov_device_vector_copy_to_host)(
    const qkrylov_device_vector_h src,
    Scalar* host_dst_complex)
{
    if (!src) {
        set_last_error("qkrylov_device_vector_copy_to_host: vector handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (src->precision != PREC_ID) {
        set_last_error("qkrylov_device_vector_copy_to_host: precision mismatch");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (!src->impl) {
        set_last_error("qkrylov_device_vector_copy_to_host: uninitialized device vector");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (src->dim > 0 && !host_dst_complex) {
        set_last_error("qkrylov_device_vector_copy_to_host: host pointer is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (src->dim == 0) return QKRYLOV_SUCCESS;

    try {
        const auto* s_vec = static_cast<const DevVector*>(src->impl.get());
        auto h_view = Kokkos::View<KComplex*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
            reinterpret_cast<KComplex*>(host_dst_complex),
            src->dim
        );
        Kokkos::deep_copy(h_view, *s_vec);
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_device_vector_copy_to_host");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int SUFFIX(qkrylov_device_vector_dot)(
    const qkrylov_device_vector_h x,
    const qkrylov_device_vector_h y,
    Scalar* dot_re,
    Scalar* dot_im)
{
    if (!x || !y) {
        set_last_error("qkrylov_device_vector_dot: vector handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (x->precision != PREC_ID || y->precision != PREC_ID) {
        set_last_error("qkrylov_device_vector_dot: precision mismatch");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (x->dim != y->dim) {
        set_last_error("qkrylov_device_vector_dot: dimension mismatch");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (!dot_re || !dot_im) {
        set_last_error("qkrylov_device_vector_dot: null output pointer");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (x->dim == 0) {
        *dot_re = Scalar(0.0);
        *dot_im = Scalar(0.0);
        return QKRYLOV_SUCCESS;
    }
    if (!x->impl || !y->impl) {
        set_last_error("qkrylov_device_vector_dot: uninitialized device vector");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        const auto* x_v = static_cast<const DevVector*>(x->impl.get());
        const auto* y_v = static_cast<const DevVector*>(y->impl.get());
        auto res = qkrylov::QKRYLOV_PRECISION_NAMESPACE::dot(*x_v, *y_v);
        *dot_re = static_cast<Scalar>(res.real());
        *dot_im = static_cast<Scalar>(res.imag());
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_device_vector_dot");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int SUFFIX(qkrylov_device_vector_norm)(
    const qkrylov_device_vector_h x,
    Scalar* norm_out)
{
    if (!x) {
        set_last_error("qkrylov_device_vector_norm: vector handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (x->precision != PREC_ID) {
        set_last_error("qkrylov_device_vector_norm: precision mismatch");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (!norm_out) {
        set_last_error("qkrylov_device_vector_norm: null output pointer");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (x->dim == 0) {
        *norm_out = Scalar(0.0);
        return QKRYLOV_SUCCESS;
    }
    if (!x->impl) {
        set_last_error("qkrylov_device_vector_norm: uninitialized device vector");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        const auto* x_v = static_cast<const DevVector*>(x->impl.get());
        auto res = qkrylov::QKRYLOV_PRECISION_NAMESPACE::norm(*x_v);
        *norm_out = static_cast<Scalar>(res);
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_device_vector_norm");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int SUFFIX(qkrylov_device_vector_axpy)(
    Scalar a_re, Scalar a_im,
    const qkrylov_device_vector_h x,
    qkrylov_device_vector_h y)
{
    if (!x || !y) {
        set_last_error("qkrylov_device_vector_axpy: vector handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (x->precision != PREC_ID || y->precision != PREC_ID) {
        set_last_error("qkrylov_device_vector_axpy: precision mismatch");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (x->dim != y->dim) {
        set_last_error("qkrylov_device_vector_axpy: dimension mismatch");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (x->dim == 0) return QKRYLOV_SUCCESS;
    if (!x->impl || !y->impl) {
        set_last_error("qkrylov_device_vector_axpy: uninitialized device vector");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        const auto* x_v = static_cast<const DevVector*>(x->impl.get());
        auto* y_v = static_cast<DevVector*>(y->impl.get());
        qkrylov::QKRYLOV_PRECISION_NAMESPACE::axpy(KComplex(a_re, a_im), *x_v, *y_v);
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_device_vector_axpy");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int SUFFIX(qkrylov_device_vector_scal)(
    Scalar a_re, Scalar a_im,
    qkrylov_device_vector_h x)
{
    if (!x) {
        set_last_error("qkrylov_device_vector_scal: vector handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (x->precision != PREC_ID) {
        set_last_error("qkrylov_device_vector_scal: precision mismatch");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (x->dim == 0) return QKRYLOV_SUCCESS;
    if (!x->impl) {
        set_last_error("qkrylov_device_vector_scal: uninitialized device vector");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        auto* x_v = static_cast<DevVector*>(x->impl.get());
        qkrylov::QKRYLOV_PRECISION_NAMESPACE::scal(KComplex(a_re, a_im), *x_v);
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_device_vector_scal");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int SUFFIX(qkrylov_device_vector_normalize)(
    qkrylov_device_vector_h x)
{
    if (!x) {
        set_last_error("qkrylov_device_vector_normalize: vector handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (x->precision != PREC_ID) {
        set_last_error("qkrylov_device_vector_normalize: precision mismatch");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (x->dim == 0) return QKRYLOV_SUCCESS;
    if (!x->impl) {
        set_last_error("qkrylov_device_vector_normalize: uninitialized device vector");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        auto* x_v = static_cast<DevVector*>(x->impl.get());
        qkrylov::QKRYLOV_PRECISION_NAMESPACE::normalize(*x_v);
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_device_vector_normalize");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int SUFFIX(qkrylov_device_vector_zero_fill)(
    qkrylov_device_vector_h x)
{
    if (!x) {
        set_last_error("qkrylov_device_vector_zero_fill: vector handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (x->precision != PREC_ID) {
        set_last_error("qkrylov_device_vector_zero_fill: precision mismatch");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (x->dim == 0) return QKRYLOV_SUCCESS;
    if (!x->impl) {
        set_last_error("qkrylov_device_vector_zero_fill: uninitialized device vector");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        auto* x_v = static_cast<DevVector*>(x->impl.get());
        qkrylov::QKRYLOV_PRECISION_NAMESPACE::zero_fill(*x_v);
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_device_vector_zero_fill");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int SUFFIX(qkrylov_device_vector_copy)(
    const qkrylov_device_vector_h src,
    qkrylov_device_vector_h dst)
{
    if (!src || !dst) {
        set_last_error("qkrylov_device_vector_copy: vector handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (src->precision != PREC_ID || dst->precision != PREC_ID) {
        set_last_error("qkrylov_device_vector_copy: precision mismatch");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (src->dim != dst->dim) {
        set_last_error("qkrylov_device_vector_copy: dimension mismatch");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (src->dim == 0) return QKRYLOV_SUCCESS;
    if (!src->impl || !dst->impl) {
        set_last_error("qkrylov_device_vector_copy: uninitialized device vector");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        const auto* s_v = static_cast<const DevVector*>(src->impl.get());
        auto* d_v = static_cast<DevVector*>(dst->impl.get());
        qkrylov::QKRYLOV_PRECISION_NAMESPACE::deep_copy(*d_v, *s_v);
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_device_vector_copy");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

void* SUFFIX(qkrylov_device_vector_data)(qkrylov_device_vector_h vec) {
    if (!vec) {
        set_last_error("qkrylov_device_vector_data: vector handle is null");
        return nullptr;
    }
    if (vec->precision != PREC_ID) {
        set_last_error("qkrylov_device_vector_data: precision mismatch");
        return nullptr;
    }
    if (!vec->impl) {
        set_last_error("qkrylov_device_vector_data: uninitialized device vector");
        return nullptr;
    }
    auto* v = static_cast<DevVector*>(vec->impl.get());
    return v ? static_cast<void*>(v->data()) : nullptr;
}

} // extern "C"
