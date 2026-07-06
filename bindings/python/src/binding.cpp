#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/shared_ptr.h>

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
#include "qkrylov/solvers/lanczos.hpp"
#include "qkrylov/solvers/davidson.hpp"
#include "qkrylov/solvers/dynamics.hpp"

namespace nb = nanobind;
using namespace nb::literals;

using namespace qkrylov;

NB_MODULE(qkrylov_cpp, m) {
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
            if (tuple.size() < 3) throw std::runtime_error("OpSum += requires at least (coeff, op, site)");
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

    nb::class_<FermionSite, Site>(m, "FermionSite")
        .def(nb::init<>());

    nb::class_<HubbardSite, Site>(m, "HubbardSite")
        .def(nb::init<>());

    nb::class_<TJSite, Site>(m, "TJSite")
        .def(nb::init<>());

    nb::class_<MatrixFreeHamiltonian>(m, "MatrixFreeHamiltonian")
        .def(nb::init<std::shared_ptr<Basis>, std::shared_ptr<Site>, const OpSum&>())
        .def("apply", &MatrixFreeHamiltonian::apply)
        .def("dimension", &MatrixFreeHamiltonian::dimension)
        .def("diagonal", &MatrixFreeHamiltonian::diagonal);

    nb::class_<LanczosResult>(m, "LanczosResult")
        .def_rw("energy", &LanczosResult::energy)
        .def_rw("eigenvector", &LanczosResult::eigenvector);

    m.def("lanczos_ground_state", &lanczos_ground_state, "H"_a, "maxiter"_a = 200, "tol"_a = 1e-12);

    nb::class_<DavidsonResult>(m, "DavidsonResult")
        .def_rw("eigenvalues", &DavidsonResult::eigenvalues)
        .def_rw("eigenvectors", &DavidsonResult::eigenvectors);

    m.def("davidson_lowest", &davidson_lowest, "H"_a, "n_eig"_a = 1, "max_subspace"_a = 20, "tol"_a = 1e-8);

    nb::class_<DynamicsResult>(m, "DynamicsResult")
        .def_rw("alphas", &DynamicsResult::alphas)
        .def_rw("betas", &DynamicsResult::betas)
        .def_rw("norm_phi0", &DynamicsResult::norm_phi0);

    m.def("continued_fraction_coeffs", &continued_fraction_coeffs, "H"_a, "phi0"_a, "n_iter"_a = 100);
    m.def("evaluate_spectral_function", &evaluate_spectral_function, "res"_a, "omega"_a, "E0"_a, "eta"_a = 0.1);
}
