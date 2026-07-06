#include <jlcxx/jlcxx.hpp>

#include "qkrylov/symmetry/sector.hpp"
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
#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"
#include "qkrylov/solvers/lanczos.hpp"

using namespace qkrylov;

JLCXX_MODULE define_julia_module(jlcxx::Module& types)
{
    types.add_type<Sector>("Sector")
        .constructor<>()
        .method("use_sz!", [](Sector& s, bool v) { s.use_sz = v; })
        .method("sz2!", [](Sector& s, int v) { s.sz2 = v; });

    types.add_type<Basis>("Basis");
    types.add_type<SpinHalfBasis>("SpinHalfBasis", jlcxx::julia_base_type<Basis>())
        .constructor<int, const Sector&>();

    types.add_type<Site>("Site");
    types.add_type<SpinHalfSite>("SpinHalfSite", jlcxx::julia_base_type<Site>())
        .constructor<>();

    types.add_type<OpSum>("OpSum")
        .constructor<>()
        .method("add_term", &OpSum::add_term);

    types.add_type<MatrixFreeHamiltonian>("MatrixFreeHamiltonian")
        .constructor<std::shared_ptr<Basis>, std::shared_ptr<Site>, const OpSum&>();

    types.method("lanczos_ground_state", &lanczos_ground_state);
}
