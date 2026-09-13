#undef NDEBUG
#include <cassert>
#include <iostream>
#include <cmath>

#include "qkrylov/solvers/policy.hpp"
#include "qkrylov/solvers/lanczos.hpp"
#include "qkrylov/symmetry/sector.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"
#include "qkrylov/basis/spin_s_basis.hpp"
#include "qkrylov/basis/fermion_basis.hpp"
#include "qkrylov/basis/hubbard_basis.hpp"
#include "qkrylov/basis/tj_basis.hpp"
#include "qkrylov/core/device.hpp"
#include "qkrylov/core/traits.hpp"
#include "qkrylov/operators/local_op.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"
#include "qkrylov/c_api.h"

using namespace qkrylov;
using namespace qkrylov::QKRYLOV_PRECISION_NAMESPACE;

void test_policy_traits() {
    std::cout << "Testing policy traits..." << std::endl;
    static_assert(solvers::policy::is_policy_v<solvers::policy::Default>);
    static_assert(solvers::policy::is_policy_v<solvers::policy::SinglePass>);
    static_assert(solvers::policy::is_policy_v<solvers::policy::TwoPass>);
    static_assert(!solvers::policy::is_policy_v<int>);
    std::cout << "Policy traits OK!" << std::endl;
}

void test_strong_sectors_and_bases() {
    std::cout << "Testing strong sectors and basis constructors..." << std::endl;
    
    // SpinHalf with strong Sz and Unconstrained
    basis::SpinHalf b_sh_sz(4, basis::sector::Sz{0});
    assert(b_sh_sz.size() == 6);
    basis::SpinHalf b_sh_un(4, basis::sector::Unconstrained{});
    assert(b_sh_un.size() == 16);

    // SpinS with strong Sz and Unconstrained
    basis::SpinS b_ss_sz(2, 0.5, basis::sector::Sz{0});
    assert(b_ss_sz.size() == 2);
    basis::SpinS b_ss_un(2, 0.5, basis::sector::Unconstrained{});
    assert(b_ss_un.size() == 4);

    // Fermion with strong Particles and Unconstrained
    basis::Fermion b_ferm_p(4, basis::sector::Particles{2});
    assert(b_ferm_p.size() == 6);
    basis::Fermion b_ferm_un(4, basis::sector::Unconstrained{});
    assert(b_ferm_un.size() == 16);

    // Hubbard with strong Hubbard and Unconstrained
    basis::Hubbard b_hub_h(2, basis::sector::Hubbard{1, 1});
    assert(b_hub_h.size() == 4);
    basis::Hubbard b_hub_un(2, basis::sector::Unconstrained{});
    assert(b_hub_un.size() == 16);

    // TJ with strong Hubbard and Unconstrained
    basis::TJ b_tj_h(2, basis::sector::Hubbard{1, 0});
    assert(b_tj_h.size() == 2);
    basis::TJ b_tj_un(2, basis::sector::Unconstrained{});
    assert(b_tj_un.size() == 9);

    // Bosons sector check
    basis::sector::Bosons bos(3);
    assert(bos.nb == 3);

    std::cout << "Strong sectors and basis constructors OK!" << std::endl;
}

void test_device_tags_and_ctad_hamiltonian() {
    std::cout << "Testing device tags and CTAD Hamiltonian..." << std::endl;

    static_assert(std::is_same_v<traits::device_traits<device::cpu>::execution_space,
                                traits::device_execution_space_t<device::cpu>>);

    auto basis = basis::SpinHalf(2, basis::sector::Sz{0});

    OpSum os;
    os += 1.0 * Sz(0) * Sz(1) + 0.5 * Sp(0) * Sm(1) + 0.5 * Sm(0) * Sp(1);

    // CTAD with site auto-inference and device::cpu
    Hamiltonian H_cpu(basis, os, device::cpu{});
    assert(H_cpu.dimension() == 2);

    // CTAD with site auto-inference and default device
    Hamiltonian H_def(basis, os);
    assert(H_def.dimension() == 2);

    // CTAD with device::gpu (maps to Cuda/HIP/SYCL or CPU fallback)
    Hamiltonian H_gpu(basis, os, device::gpu{});
    assert(H_gpu.dimension() == 2);

    std::cout << "Device tags and CTAD Hamiltonian OK!" << std::endl;
}

void test_solvers_structured_bindings() {
    std::cout << "Testing policy dispatch and structured bindings..." << std::endl;

    auto basis = basis::SpinHalf(2, basis::sector::Unconstrained{});
    OpSum os;
    os += 1.0 * Sz(0) * Sz(1) + 0.5 * Sp(0) * Sm(1) + 0.5 * Sm(0) * Sp(1);

    Hamiltonian H(basis, os, device::cpu{});
    LanczosConfig config{100, 1e-12};

    // Default policy structured binding
    auto [e_def, v_def] = solvers::lanczos(H, config);
    assert(std::abs(e_def - (-0.75)) < 1e-6);
    assert(v_def.size() == 4);

    // SinglePass policy structured binding
    auto [e_sp, v_sp] = solvers::lanczos<solvers::policy::SinglePass>(H, config);
    assert(std::abs(e_sp - (-0.75)) < 1e-6);
    assert(v_sp.size() == 4);

    // TwoPass policy structured binding
    auto [e_tp, v_tp] = solvers::lanczos<solvers::policy::TwoPass>(H, config);
    assert(std::abs(e_tp - (-0.75)) < 1e-5);
    assert(v_tp.size() == 4);

    const Real comp_tol = (sizeof(Real) == 4) ? Real(1e-5) : Real(1e-10);
    assert(std::abs(e_sp - e_tp) < comp_tol);

    // Convenience zero-flag functions
    auto res_gs = lanczos_ground_state(H, 100, 1e-6);
    assert(std::abs(res_gs.energy - (-0.75)) < 1e-5);
    assert(res_gs.eigenvector.size() == 4);

    auto res_tp_conv = lanczos_two_pass(H, 100, 1e-6);
    assert(std::abs(res_tp_conv.energy - (-0.75)) < 1e-5);
    assert(res_tp_conv.eigenvector.size() == 4);

    assert(std::abs(res_gs.energy - res_tp_conv.energy) < comp_tol);

    // Verify reference structured bindings and mutation semantics
    LanczosResult res = solvers::lanczos(H, config);
    static_assert(std::is_same_v<decltype(res.get<0>()), Real&>);
    static_assert(std::is_same_v<decltype(res.get<1>()), HostVector&>);

    auto& [e_ref, v_ref] = res;
    assert(&e_ref == &res.energy);
    assert(&v_ref == &res.eigenvector);

    // Mutation via reference modifies underlying LanczosResult
    e_ref = -42.0;
    assert(res.energy == -42.0);
    v_ref[0] = Complex(99.0, 0.0);
    assert(res.eigenvector[0] == Complex(99.0, 0.0));

    // Const reference structured binding
    const auto& [e_cref, v_cref] = res;
    assert(&e_cref == &res.energy);
    assert(&v_cref == &res.eigenvector);
    assert(e_cref == -42.0);

    std::cout << "Solvers policy dispatch and structured bindings OK!" << std::endl;
}

void test_c_api_endpoints() {
    std::cout << "Testing C ABI zero-flag endpoints..." << std::endl;

    qkrylov_basis_h b = qkrylov_spinhalf_basis_create(2, nullptr);
    assert(b != nullptr);
    qkrylov_site_h s = qkrylov_spinhalf_site_create();
    assert(s != nullptr);
    qkrylov_opsum_h ops = qkrylov_opsum_create();
    assert(ops != nullptr);
    assert(qkrylov_opsum_add_term_2body(ops, 1.0, 0.0, "Sz", 0, "Sz", 1) == QKRYLOV_SUCCESS);
    assert(qkrylov_opsum_add_term_2body(ops, 0.5, 0.0, "Sp", 0, "Sm", 1) == QKRYLOV_SUCCESS);
    assert(qkrylov_opsum_add_term_2body(ops, 0.5, 0.0, "Sm", 0, "Sp", 1) == QKRYLOV_SUCCESS);

    qkrylov_hamiltonian_h H = qkrylov_hamiltonian_create(b, s, ops);
    assert(H != nullptr);

    // Single pass FP64
    qkrylov_lanczos_result_fp64_t r_sp;
    assert(qkrylov_lanczos_ground_state_fp64(H, 100, 1e-12, &r_sp) == QKRYLOV_SUCCESS);
    assert(std::abs(r_sp.energy - (-0.75)) < 1e-6);

    // Two pass FP64
    qkrylov_lanczos_result_fp64_t r_tp;
    assert(qkrylov_lanczos_two_pass_ground_state_fp64(H, 100, 1e-12, &r_tp) == QKRYLOV_SUCCESS);
    assert(std::abs(r_tp.energy - (-0.75)) < 1e-6);

    assert(std::abs(r_sp.energy - r_tp.energy) < 1e-10);

    qkrylov_hamiltonian_destroy(H);
    qkrylov_opsum_destroy(ops);
    qkrylov_site_destroy(s);
    qkrylov_basis_destroy(b);

    std::cout << "C ABI zero-flag endpoints OK!" << std::endl;
}

int main() {
    test_policy_traits();
    test_strong_sectors_and_bases();
    test_device_tags_and_ctad_hamiltonian();
    test_solvers_structured_bindings();
    test_c_api_endpoints();

    std::cout << "All M1 core tests passed successfully!" << std::endl;
    return 0;
}
