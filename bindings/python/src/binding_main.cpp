#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "qkrylov/symmetry/sector.hpp"
#include "qkrylov/core/device.hpp"
#include "qkrylov/basis/basis.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"
#include "qkrylov/basis/fermion_basis.hpp"
#include "qkrylov/basis/hubbard_basis.hpp"
#include "qkrylov/basis/tj_basis.hpp"
#include "qkrylov/basis/spin_s_basis.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace qkrylov;

static void bind_common(nb::module_& m) {
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

    nb::class_<Device>(m, "Device")
        .def(nb::init<>())
        .def(nb::init<const std::string&>(), "device_string"_a)
        .def(nb::init<int>(), "device_id"_a)
        .def_ro("id", &Device::id)
        .def_static("is_gpu_build", &Device::is_gpu_build)
        .def_static("backend_name", &Device::backend_name)
        .def_static("gpu_count", &Device::gpu_count);

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

    nb::class_<SpinSBasis, Basis>(m, "SpinSBasis")
        .def(nb::init<int, double, const Sector&>(), "N"_a, "S"_a = 0.5, "sector"_a = Sector())
        .def("size", &SpinSBasis::size)
        .def("state", &SpinSBasis::state)
        .def("index", &SpinSBasis::index)
        .def("contains", &SpinSBasis::contains)
        .def("nsites", &SpinSBasis::nsites)
        .def_prop_ro("spin", &SpinSBasis::spin)
        .def_prop_ro("dimension_per_site", &SpinSBasis::dimension_per_site);

    const char* names[] = {
        "Sector", "Device", "Basis", "SpinHalfBasis", "FermionBasis", "HubbardBasis", "TJBasis", "SpinSBasis"
    };
    for (const char* name : names) {
        m.attr((std::string(name) + "_FP32").c_str()) = m.attr(name);
        m.attr((std::string(name) + "_FP64").c_str()) = m.attr(name);
    }
}

void bind_fp32(nb::module_& m);
void bind_fp64(nb::module_& m);

NB_MODULE(_qkrylov_cpp, m) {
    bind_common(m);
    bind_fp32(m);
    bind_fp64(m);
}
