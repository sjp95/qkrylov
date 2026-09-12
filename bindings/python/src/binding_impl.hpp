#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/ndarray.h>

#include "qkrylov/symmetry/sector.hpp"
#include "qkrylov/operators/operator_term.hpp"
#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/basis/basis.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"
#include "qkrylov/basis/fermion_basis.hpp"
#include "qkrylov/basis/hubbard_basis.hpp"
#include "qkrylov/basis/tj_basis.hpp"
#include "qkrylov/sites/site.hpp"
#include "qkrylov/sites/spinhalf_site.hpp"
#include "qkrylov/sites/fermion_site.hpp"
#include "qkrylov/sites/hubbard_site.hpp"
#include "qkrylov/sites/tj_site.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"
#include "qkrylov/core/device.hpp"
#include "qkrylov/solvers/lanczos.hpp"
#include "qkrylov/solvers/policy.hpp"
#include "qkrylov/solvers/davidson.hpp"
#include "qkrylov/solvers/dynamics.hpp"
#include "qkrylov/solvers/ftlm.hpp"
#include "qkrylov/basis/spin_s_basis.hpp"
#include "qkrylov/sites/spin_s_site.hpp"
#include "qkrylov/solvers/correction_vector.hpp"

namespace nb = nanobind;
using namespace nb::literals;

using namespace qkrylov;
using namespace qkrylov::QKRYLOV_PRECISION_NAMESPACE;

// For compatibility with any headers that might define HostVector
using HostVector = std::vector<Complex>;

// Convenience alias for a 1-D complex128 C-contiguous ndarray (read-only view)
using CxArray = nb::ndarray<const Complex, nb::shape<-1>, nb::c_contig, nb::device::cpu>;

// Helper: wrap an existing std::vector<Complex> as a zero-copy NumPy array.
// The returned ndarray keeps the vector alive via a capsule.
static nb::ndarray<nb::numpy, Complex, nb::shape<-1>>
vec_to_numpy(std::vector<Complex>&& v)
{
    auto data = std::make_unique<std::vector<Complex>>(std::move(v));
    auto* raw_ptr = data.get();
    nb::capsule owner(raw_ptr, [](void* p) noexcept {
        delete static_cast<std::vector<Complex>*>(p);
    });
    data.release();
    return nb::ndarray<nb::numpy, Complex, nb::shape<-1>>(
        raw_ptr->data(), { raw_ptr->size() }, owner
    );
}

// Same for Real vectors (e.g. alphas/betas in dynamics)
static nb::ndarray<nb::numpy, Real, nb::shape<-1>>
dvec_to_numpy(std::vector<Real>&& v)
{
    auto data = std::make_unique<std::vector<Real>>(std::move(v));
    auto* raw_ptr = data.get();
    nb::capsule owner(raw_ptr, [](void* p) noexcept {
        delete static_cast<std::vector<Real>*>(p);
    });
    data.release();
    return nb::ndarray<nb::numpy, Real, nb::shape<-1>>(
        raw_ptr->data(), { raw_ptr->size() }, owner
    );
}


template <typename ExecSpace>
static void bind_backend(nb::module_& m, const std::string& suffix, const std::string& type_suffix) {
    using HType = MatrixFreeHamiltonian<ExecSpace>;
    
    std::string h_name = "MatrixFreeHamiltonian" + suffix + type_suffix;
    nb::class_<HType>(m, h_name.c_str())
        .def(nb::init<std::shared_ptr<Basis>, std::shared_ptr<Site>, const OpSum&, Device>(),
             "basis"_a, "site"_a, "ops"_a, "device"_a = Device())
        .def("apply", [](const HType& H, CxArray x) {
            const Index n = H.dimension();
            if (x.shape(0) != static_cast<size_t>(n)) {
                throw std::invalid_argument("Input array size does not match Hamiltonian dimension");
            }
            HostVector y_vec(n);
            H.apply(x.data(), y_vec.data());
            return vec_to_numpy(std::move(y_vec));
        })
        .def("dimension", &HType::dimension)
        .def("diagonal", [](const HType& H) {
            return vec_to_numpy(H.diagonal_host());
        });

    std::string lgs_name = "lanczos_ground_state_" + suffix + type_suffix;
    m.def(lgs_name.c_str(),
        [](const HType& H, int maxiter, Real tol) -> nb::tuple {
            LanczosConfig cfg;
            cfg.maxiter = maxiter;
            cfg.tol = tol;
            auto res = solvers::lanczos<solvers::policy::OnePass_full>(H, cfg);
            return nb::make_tuple(res.energy, vec_to_numpy(std::move(res.eigenvector)));
        },
        "H"_a, "maxiter"_a = 200, "tol"_a = 1e-12);

    std::string ltp_name = "lanczos_two_pass_" + suffix + type_suffix;
    m.def(ltp_name.c_str(),
        [](const HType& H, int maxiter, Real tol) -> nb::tuple {
            LanczosConfig cfg;
            cfg.maxiter = maxiter;
            cfg.tol = tol;
            auto res = solvers::lanczos<solvers::policy::TwoPass>(H, cfg);
            return nb::make_tuple(res.energy, vec_to_numpy(std::move(res.eigenvector)));
        },
        "H"_a, "maxiter"_a = 200, "tol"_a = 1e-12);

    std::string dav_name = "davidson_lowest_" + suffix + type_suffix;
    m.def(dav_name.c_str(), &davidson_lowest<ExecSpace>,
          "H"_a, "n_eig"_a = 1, "max_subspace"_a = 20, "tol"_a = 1e-8);

    std::string dyn_name = "continued_fraction_coeffs_" + suffix + type_suffix;
    m.def(dyn_name.c_str(),
        [](const HType& H, CxArray phi0, int n_iter) {
            if (phi0.shape(0) != static_cast<size_t>(H.dimension())) {
                throw std::invalid_argument("phi0 vector size does not match Hamiltonian dimension");
            }
            const HostVector phi0_vec(phi0.data(), phi0.data() + phi0.shape(0));
            auto res = continued_fraction_coeffs<ExecSpace>(H, phi0_vec, n_iter);
            return nb::make_tuple(
                dvec_to_numpy(std::move(res.alphas)),
                dvec_to_numpy(std::move(res.betas)),
                res.norm_phi0
            );
        },
        "H"_a, "phi0"_a, "n_iter"_a = 100);

    std::string ftlm_name = "ftlm_" + suffix + type_suffix;
    m.def(ftlm_name.c_str(), [](const HType& H, Real beta, int n_random, int n_steps) {
        return ftlm<ExecSpace>(H, beta, n_random, n_steps);
    }, "H"_a, "beta"_a, "n_random"_a = 50, "n_steps"_a = 100);

    std::string ftlm_sweep_name = "ftlm_sweep_" + suffix + type_suffix;
    m.def(ftlm_sweep_name.c_str(),
        [](const HType& H, const std::vector<Real>& beta_grid,
           const std::vector<HType>& observables,
           int n_random, int n_steps, uint64_t seed) {
            return ftlm_sweep<ExecSpace>(H, beta_grid, observables, n_random, n_steps, seed);
        },
        "H"_a, "beta_grid"_a, "observables"_a = std::vector<HType>{},
        "n_random"_a = 50, "n_steps"_a = 100, "seed"_a = 42);

    std::string cv_name = "correction_vector_spectral_" + suffix + type_suffix;
    m.def(cv_name.c_str(),
        [](const HType& H, CxArray op_psi0, Real E0, Real omega, Real eta, int max_iter, Real tol) {
            if (op_psi0.shape(0) != static_cast<size_t>(H.dimension())) {
                throw std::invalid_argument("op_psi0 vector size does not match Hamiltonian dimension");
            }
            const HostVector op_psi0_vec(op_psi0.data(), op_psi0.data() + op_psi0.shape(0));
            auto res = correction_vector_spectral<ExecSpace>(H, op_psi0_vec, E0, omega, eta, max_iter, tol);
            return nb::make_tuple(
                vec_to_numpy(std::move(res.correction_vector)),
                res.spectral_function,
                res.iterations,
                res.converged
            );
        },
        "H"_a, "op_psi0"_a, "E0"_a, "omega"_a, "eta"_a = static_cast<Real>(0.1), "max_iter"_a = 500, "tol"_a = static_cast<Real>(1e-8));
}

static void bind_impl(nb::module_& m, const std::string& type_suffix) {
    nb::class_<OperatorFactor>(m, ("OperatorFactor" + type_suffix).c_str())
        .def(nb::init<std::string, int>(), "op"_a, "site"_a)
        .def_rw("op", &OperatorFactor::op)
        .def_rw("site", &OperatorFactor::site);

    nb::class_<OperatorTerm>(m, ("OperatorTerm" + type_suffix).c_str())
        .def(nb::init<>())
        .def_rw("coeff", &OperatorTerm::coeff)
        .def_rw("factors", &OperatorTerm::factors);

    nb::class_<OpSum>(m, ("OpSum" + type_suffix).c_str())
        .def(nb::init<>())
        .def("add_term", &OpSum::add_term)
        .def("__iadd__", [](OpSum& os, nb::tuple tuple) {
            if (tuple.size() < 3 || tuple.size() % 2 == 0) {
                throw std::invalid_argument("OpSum += requires (coeff, op1, site1, [op2, site2, ...]) with odd tuple length >= 3");
            }
            OperatorTerm term;
            term.coeff = nb::cast<Complex>(tuple[0]);
            for (size_t i = 1; i < tuple.size(); i += 2) {
                term.factors.push_back({nb::cast<std::string>(tuple[i]), nb::cast<int>(tuple[i+1])});
            }
            os.add_term(term);
            return &os;
        })
        .def("clear", &OpSum::clear)
        .def("size", &OpSum::size)
        .def("terms", &OpSum::terms);

    nb::class_<Site>(m, ("Site" + type_suffix).c_str());

    nb::class_<SpinHalfSite, Site>(m, ("SpinHalfSite" + type_suffix).c_str())
        .def(nb::init<>());

    nb::class_<SpinSSite, Site>(m, ("SpinSSite" + type_suffix).c_str())
        .def(nb::init<double>(), "S"_a = 0.5)
        .def_prop_ro("spin", &SpinSSite::spin)
        .def_prop_ro("dimension_per_site", &SpinSSite::dimension_per_site);

    nb::class_<FermionSite, Site>(m, ("FermionSite" + type_suffix).c_str())
        .def(nb::init<>());

    nb::class_<HubbardSite, Site>(m, ("HubbardSite" + type_suffix).c_str())
        .def(nb::init<>());

    nb::class_<TJSite, Site>(m, ("TJSite" + type_suffix).c_str())
        .def(nb::init<>());

    nb::class_<CorrectionVectorResult>(m, ("CorrectionVectorResult" + type_suffix).c_str())
        .def(nb::init<>())
        .def_ro("spectral_function", &CorrectionVectorResult::spectral_function)
        .def_ro("iterations", &CorrectionVectorResult::iterations)
        .def_ro("converged", &CorrectionVectorResult::converged)
        .def_prop_ro("correction_vector", [](const CorrectionVectorResult& self) {
            std::vector<Complex> copy = self.correction_vector;
            return vec_to_numpy(std::move(copy));
        });

    nb::class_<DavidsonResult>(m, ("DavidsonResult" + type_suffix).c_str())
        .def_rw("eigenvalues", &DavidsonResult::eigenvalues)
        .def_rw("eigenvectors", &DavidsonResult::eigenvectors);

    nb::class_<FTLMResult>(m, ("FTLMResult" + type_suffix).c_str())
        .def_rw("beta", &FTLMResult::beta)
        .def_rw("partition_function", &FTLMResult::partition_function)
        .def_rw("free_energy", &FTLMResult::free_energy)
        .def_rw("internal_energy", &FTLMResult::internal_energy)
        .def_rw("specific_heat", &FTLMResult::specific_heat)
        .def_rw("entropy", &FTLMResult::entropy)
        .def_rw("observable_expectations", &FTLMResult::observable_expectations)
        .def_rw("observable_errors", &FTLMResult::observable_errors);

    nb::class_<FTLMSweepResult>(m, ("FTLMSweepResult" + type_suffix).c_str())
        .def_rw("beta_grid", &FTLMSweepResult::beta_grid)
        .def_rw("partition_functions", &FTLMSweepResult::partition_functions)
        .def_rw("free_energies", &FTLMSweepResult::free_energies)
        .def_rw("internal_energies", &FTLMSweepResult::internal_energies)
        .def_rw("specific_heats", &FTLMSweepResult::specific_heats)
        .def_rw("entropies", &FTLMSweepResult::entropies)
        .def_rw("observable_expectations", &FTLMSweepResult::observable_expectations)
        .def_rw("observable_errors", &FTLMSweepResult::observable_errors);

    using DblArray = nb::ndarray<const Real, nb::shape<-1>, nb::c_contig, nb::device::cpu>;
    m.def(("evaluate_spectral_function" + type_suffix).c_str(),
        [](DblArray alphas, DblArray betas, Real norm_phi0,
           Real omega, Real E0, Real eta) {
            if (alphas.shape(0) != betas.shape(0) && alphas.shape(0) != betas.shape(0) + 1) {
                throw std::invalid_argument("alphas array length must match betas array length or betas array length + 1");
            }
            return evaluate_spectral_function(
                alphas.data(), betas.data(), alphas.shape(0),
                norm_phi0, omega, E0, eta
            );
        },
        "alphas"_a, "betas"_a, "norm_phi0"_a, "omega"_a, "E0"_a, "eta"_a = 0.1);

#ifdef KOKKOS_ENABLE_SERIAL
    bind_backend<Kokkos::Serial>(m, "Serial", type_suffix);
#endif
#ifdef KOKKOS_ENABLE_OPENMP
    bind_backend<Kokkos::OpenMP>(m, "CPU", type_suffix);
#endif
#ifdef KOKKOS_ENABLE_THREADS
    bind_backend<Kokkos::Threads>(m, "Threads", type_suffix);
#endif
#ifdef KOKKOS_ENABLE_CUDA
    bind_backend<Kokkos::Cuda>(m, "CUDA", type_suffix);
#endif
#ifdef KOKKOS_ENABLE_HIP
    bind_backend<Kokkos::HIP>(m, "HIP", type_suffix);
#endif
#ifdef KOKKOS_ENABLE_SYCL
    bind_backend<Kokkos::Experimental::SYCL>(m, "SYCL", type_suffix);
#endif

    m.def(("initialize" + type_suffix).c_str(), [](const std::string& device) {
        detail::initialize_kokkos(Device(device));
    }, "device"_a = "cpu");

    m.def(("backend" + type_suffix).c_str(), &Device::backend_name);
    m.def(("is_gpu_build" + type_suffix).c_str(), &Device::is_gpu_build);
}
