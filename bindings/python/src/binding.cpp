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
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"
#ifdef QKRYLOV_ENABLE_CUDA
#include "qkrylov/hamiltonian/cuda_hamiltonian.hpp"
#endif
#include "qkrylov/solvers/lanczos.hpp"
#include "qkrylov/solvers/davidson.hpp"
#include "qkrylov/solvers/dynamics.hpp"
#include "qkrylov/solvers/correction_vector.hpp"
#include "qkrylov/solvers/ftlm.hpp"

namespace nb = nanobind;
using namespace nb::literals;

using namespace qkrylov;

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

// Same for double vectors (e.g. alphas/betas in dynamics)
static nb::ndarray<nb::numpy, double, nb::shape<-1>>
dvec_to_numpy(std::vector<double>&& v)
{
    auto data = std::make_unique<std::vector<double>>(std::move(v));
    auto* raw_ptr = data.get();
    nb::capsule owner(raw_ptr, [](void* p) noexcept {
        delete static_cast<std::vector<double>*>(p);
    });
    data.release();
    return nb::ndarray<nb::numpy, double, nb::shape<-1>>(
        raw_ptr->data(), { raw_ptr->size() }, owner
    );
}

NB_MODULE(_qkrylov_cpp, m) {
    nb::class_<Sector>(m, "Sector")
        .def(nb::init<>())
        .def_rw("use_sz", &Sector::use_sz)
        .def_rw("sz2", &Sector::sz2)
        .def_rw("use_nup", &Sector::use_nup)
        .def_rw("use_ndn", &Sector::use_ndn)
        .def_rw("nup", &Sector::nup)
        .def_rw("ndn", &Sector::ndn)
        .def_rw("use_n", &Sector::use_n)
        .def_rw("n", &Sector::n)
        .def_rw("use_nb", &Sector::use_nb)
        .def_rw("nb", &Sector::nb);

    nb::class_<OperatorFactor>(m, "OperatorFactor")
        .def(nb::init<std::string, int>(), "op"_a, "site"_a)
        .def_rw("op", &OperatorFactor::op)
        .def_rw("site", &OperatorFactor::site);

    nb::class_<OperatorTerm>(m, "OperatorTerm")
        .def(nb::init<>())
        .def_rw("coeff", &OperatorTerm::coeff)
        .def_rw("factors", &OperatorTerm::factors);

    nb::class_<OpSum>(m, "OpSum")
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

    nb::class_<Basis>(m, "Basis");

    nb::class_<SpinHalfBasis, Basis>(m, "SpinHalfBasis")
        .def(nb::init<int, const Sector&>(), "N"_a, "sector"_a = Sector())
        .def("size", &SpinHalfBasis::size)
        .def("state", &SpinHalfBasis::state)
        .def("index", &SpinHalfBasis::index)
        .def("contains", &SpinHalfBasis::contains)
        .def("nsites", &SpinHalfBasis::nsites);

    nb::class_<SpinSBasis, Basis>(m, "SpinSBasis")
        .def(nb::init<int, double, const Sector&>(), "N"_a, "S"_a = 0.5, "sector"_a = Sector())
        .def("size", &SpinSBasis::size)
        .def("state", &SpinSBasis::state)
        .def("index", &SpinSBasis::index)
        .def("contains", &SpinSBasis::contains)
        .def("nsites", &SpinSBasis::nsites);

    nb::class_<FermionBasis, Basis>(m, "FermionBasis")
        .def(nb::init<int, const Sector&>(), "N"_a, "sector"_a = Sector())
        .def("size", &FermionBasis::size)
        .def("state", &FermionBasis::state)
        .def("index", &FermionBasis::index)
        .def("contains", &FermionBasis::contains)
        .def("nsites", &FermionBasis::nsites);

    nb::class_<HubbardBasis, Basis>(m, "HubbardBasis")
        .def(nb::init<int, const Sector&>(), "N"_a, "sector"_a = Sector())
        .def("size", &HubbardBasis::size)
        .def("state", &HubbardBasis::state)
        .def("index", &HubbardBasis::index)
        .def("contains", &HubbardBasis::contains)
        .def("nsites", &HubbardBasis::nsites);

    nb::class_<TJBasis, Basis>(m, "TJBasis")
        .def(nb::init<int, const Sector&>(), "N"_a, "sector"_a = Sector())
        .def("size", &TJBasis::size)
        .def("state", &TJBasis::state)
        .def("index", &TJBasis::index)
        .def("contains", &TJBasis::contains)
        .def("nsites", &TJBasis::nsites);

    nb::class_<Site>(m, "Site");

    nb::class_<SpinHalfSite, Site>(m, "SpinHalfSite")
        .def(nb::init<>());

    nb::class_<SpinSSite, Site>(m, "SpinSSite")
        .def(nb::init<double>(), "S"_a = 0.5)
        .def("spin", &SpinSSite::spin)
        .def("dimension_per_site", &SpinSSite::dimension_per_site);

    nb::class_<FermionSite, Site>(m, "FermionSite")
        .def(nb::init<>());

    nb::class_<HubbardSite, Site>(m, "HubbardSite")
        .def(nb::init<>());

    nb::class_<TJSite, Site>(m, "TJSite")
        .def(nb::init<>());

    nb::class_<MatrixFreeHamiltonian>(m, "MatrixFreeHamiltonian")
        .def(nb::init<std::shared_ptr<Basis>, std::shared_ptr<Site>, const OpSum&>())
        .def("apply", [](const MatrixFreeHamiltonian& H, CxArray x) {
            const Index n = H.dimension();
            if (x.shape(0) != static_cast<size_t>(n)) {
                throw std::invalid_argument("Input array size (" + std::to_string(x.shape(0)) +
                                           ") does not match Hamiltonian dimension (" + std::to_string(n) + ")");
            }
            MatrixFreeHamiltonian::Vector y_vec(n);
            H.apply(x.data(), y_vec.data());
            return vec_to_numpy(std::move(y_vec));
        })
        .def("dimension", &MatrixFreeHamiltonian::dimension)
        .def("diagonal", [](const MatrixFreeHamiltonian& H) {
            return vec_to_numpy(H.diagonal());
        });

#ifdef QKRYLOV_ENABLE_CUDA
    nb::class_<CUDAHamiltonian>(m, "CUDAHamiltonian")
        .def(nb::init<std::shared_ptr<Basis>, std::shared_ptr<Site>, const OpSum&>())
        .def("apply", [](const CUDAHamiltonian& H, CxArray x) {
            const Index n = H.dimension();
            const MatrixFreeHamiltonian::Vector x_vec(x.data(), x.data() + n);
            MatrixFreeHamiltonian::Vector y_vec;
            H.apply(x_vec, y_vec);
            return vec_to_numpy(std::move(y_vec));
        })
        .def("dimension", &CUDAHamiltonian::dimension);
#endif

    nb::class_<LanczosResult>(m, "LanczosResult")
        .def_rw("energy", &LanczosResult::energy)
        .def_rw("eigenvector", &LanczosResult::eigenvector);

    m.def("lanczos_ground_state",
        [](const MatrixFreeHamiltonian& H, int maxiter, double tol) {
            auto res = lanczos_ground_state(H, maxiter, tol);
            return nb::make_tuple(res.energy, vec_to_numpy(std::move(res.eigenvector)));
        },
        "H"_a, "maxiter"_a = 200, "tol"_a = 1e-12);

    nb::class_<DavidsonResult>(m, "DavidsonResult")
        .def_rw("eigenvalues", &DavidsonResult::eigenvalues)
        .def_rw("eigenvectors", &DavidsonResult::eigenvectors);

    m.def("davidson_lowest", &davidson_lowest,
          "H"_a, "n_eig"_a = 1, "max_subspace"_a = 20, "tol"_a = 1e-8);

    nb::class_<DynamicsResult>(m, "DynamicsResult")
        .def_rw("norm_phi0", &DynamicsResult::norm_phi0)
        .def_rw("alphas", &DynamicsResult::alphas)
        .def_rw("betas", &DynamicsResult::betas);

    m.def("continued_fraction_coeffs",
        [](const MatrixFreeHamiltonian& H, CxArray phi0, int n_iter) {
            if (phi0.shape(0) != static_cast<size_t>(H.dimension())) {
                throw std::invalid_argument("phi0 vector size (" + std::to_string(phi0.shape(0)) +
                                           ") does not match Hamiltonian dimension (" + std::to_string(H.dimension()) + ")");
            }
            const Vector phi0_vec(phi0.data(), phi0.data() + phi0.shape(0));
            auto res = continued_fraction_coeffs(H, phi0_vec, n_iter);
            return nb::make_tuple(
                dvec_to_numpy(std::move(res.alphas)),
                dvec_to_numpy(std::move(res.betas)),
                res.norm_phi0
            );
        },
        "H"_a, "phi0"_a, "n_iter"_a = 100);

    using DblArray = nb::ndarray<const double, nb::shape<-1>, nb::c_contig, nb::device::cpu>;
    m.def("evaluate_spectral_function",
        [](DblArray alphas, DblArray betas, double norm_phi0,
           double omega, double E0, double eta) {
            if (alphas.shape(0) != betas.shape(0) && alphas.shape(0) != betas.shape(0) + 1) {
                throw std::invalid_argument("alphas array length must match betas array length or betas array length + 1");
            }
            return evaluate_spectral_function(
                alphas.data(), betas.data(), alphas.shape(0),
                norm_phi0, omega, E0, eta
            );
        },
        "alphas"_a, "betas"_a, "norm_phi0"_a, "omega"_a, "E0"_a, "eta"_a = 0.1);

    nb::class_<CorrectionVectorResult>(m, "CorrectionVectorResult")
        .def_rw("spectral_function", &CorrectionVectorResult::spectral_function)
        .def_rw("iterations", &CorrectionVectorResult::iterations)
        .def_rw("converged", &CorrectionVectorResult::converged);

    m.def("correction_vector_spectral",
        [](const MatrixFreeHamiltonian& H, CxArray Op_psi0, double E0, double omega, double eta, int max_iter, double tol) {
            if (Op_psi0.shape(0) != static_cast<size_t>(H.dimension())) {
                throw std::invalid_argument("Op_psi0 vector size does not match Hamiltonian dimension");
            }
            const Vector Op_psi0_vec(Op_psi0.data(), Op_psi0.data() + Op_psi0.shape(0));
            auto res = correction_vector_spectral(H, Op_psi0_vec, E0, omega, eta, max_iter, tol);
            return nb::make_tuple(vec_to_numpy(std::move(res.correction_vector)), res.spectral_function, res.iterations, res.converged);
        },
        "H"_a, "Op_psi0"_a, "E0"_a, "omega"_a, "eta"_a = 0.1, "max_iter"_a = 500, "tol"_a = 1e-8);

    nb::class_<FTLMResult>(m, "FTLMResult")
        .def_rw("beta", &FTLMResult::beta)
        .def_rw("partition_function", &FTLMResult::partition_function)
        .def_rw("internal_energy", &FTLMResult::internal_energy)
        .def_rw("specific_heat", &FTLMResult::specific_heat);

    m.def("ftlm", &ftlm,
          "H"_a, "beta"_a, "n_random"_a = 50, "n_steps"_a = 100);
}
