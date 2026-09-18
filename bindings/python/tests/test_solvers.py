import math
import numpy as np
import pytest

import qkrylov as qk
from qkrylov.solvers import (
    Solver,
    Lanczos,
    LanczosTwoPass,
    Davidson,
    FTLM,
    ContinuedFraction,
    CorrectionVector,
    LanczosResult,
    DavidsonResult,
    DynamicsResult,
    FTLMResult,
    FTLMSweepResult,
    CorrectionVectorResult,
)


def _build_heisenberg_model(N=4, dtype=np.float64):
    basis = qk.basis.SpinHalf(N=N, sz=0, dtype=dtype)
    site = qk.site.SpinHalf(dtype=dtype)
    os = qk.OpSum()
    for i in range(N - 1):
        os += 1.0 * qk.Sz(i) * qk.Sz(i + 1) + 0.5 * (qk.Sp(i) * qk.Sm(i + 1) + qk.Sm(i) * qk.Sp(i + 1))
    H = qk.MatrixFreeHamiltonian(basis, site, os, dtype=dtype)
    return basis, site, os, H


def test_solver_hierarchy():
    assert issubclass(Lanczos, Solver)
    assert issubclass(LanczosTwoPass, Solver)
    assert issubclass(Davidson, Solver)
    assert issubclass(FTLM, Solver)
    assert issubclass(ContinuedFraction, Solver)
    assert issubclass(CorrectionVector, Solver)


def test_lanczos_parameters():
    s1 = Lanczos(maxiter=150, tol=1e-10)
    assert s1.maxiter == 150
    assert s1.max_iter == 150
    assert s1.tol == 1e-10

    s2 = Lanczos(max_iter=300, tol=1e-8)
    assert s2.maxiter == 300
    assert s2.max_iter == 300

    s3 = LanczosTwoPass(maxiter=250, tol=1e-9)
    assert s3.maxiter == 250
    assert s3.max_iter == 250
    assert s3.tol == 1e-9

    s4 = LanczosTwoPass(max_iter=400, tol=1e-11)
    assert s4.maxiter == 400
    assert s4.max_iter == 400


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_lanczos_single_pass_and_two_pass_equivalence(dtype):
    _, _, _, H = _build_heisenberg_model(N=4, dtype=dtype)
    dim = H.dimension
    assert dim == 6

    tol = 1e-6 if dtype == np.float32 else 1e-12
    exact_e0 = -1.6160254038

    # 1. SinglePass
    solver_sp = Lanczos(maxiter=200, tol=tol)
    res_sp = solver_sp.solve(H)
    assert isinstance(res_sp, LanczosResult)
    assert math.isclose(res_sp.energy, exact_e0, abs_tol=1e-4 if dtype == np.float32 else 1e-5)
    assert len(res_sp.eigenvector) == dim

    # Tuple unpacking
    e_sp, psi_sp = solver_sp.solve(H)
    assert math.isclose(e_sp, res_sp.energy, abs_tol=1e-12)
    assert np.allclose(psi_sp, res_sp.eigenvector)

    # 2. TwoPass
    solver_tp = LanczosTwoPass(maxiter=200, tol=tol)
    res_tp = solver_tp.solve(H)
    assert isinstance(res_tp, LanczosResult)
    assert math.isclose(res_tp.energy, exact_e0, abs_tol=1e-4 if dtype == np.float32 else 1e-5)
    assert len(res_tp.eigenvector) == dim

    # Tuple unpacking
    e_tp, psi_tp = solver_tp.solve(H)
    assert math.isclose(e_tp, res_tp.energy, abs_tol=1e-12)
    assert np.allclose(psi_tp, res_tp.eigenvector)

    # 3. Equivalence
    energy_tol = 1e-5 if dtype == np.float32 else 1e-9
    assert math.isclose(res_sp.energy, res_tp.energy, abs_tol=energy_tol)

    # 4. Check Rayleigh quotient H @ psi == E0 * psi
    y_sp = H @ psi_sp
    norm_sp = np.linalg.norm(psi_sp)
    rayleigh_sp = np.vdot(psi_sp, y_sp).real / (norm_sp ** 2)
    assert math.isclose(rayleigh_sp, res_sp.energy, abs_tol=energy_tol)

    y_tp = H @ psi_tp
    norm_tp = np.linalg.norm(psi_tp)
    rayleigh_tp = np.vdot(psi_tp, y_tp).real / (norm_tp ** 2)
    assert math.isclose(rayleigh_tp, res_tp.energy, abs_tol=energy_tol)


def test_solver_callable_syntax():
    _, _, _, H = _build_heisenberg_model(N=4, dtype=np.float64)
    solver = qk.solvers.Lanczos(maxiter=100, tol=1e-10)
    # Call solver as a function: solver(H)
    evals, evecs = solver(H)
    assert math.isclose(evals, -1.6160254038, abs_tol=1e-5)
    assert len(evecs) == 6


def test_result_tuple_protocols():
    # LanczosResult protocol
    v = np.array([1.0, 0.0], dtype=np.complex128)
    lr = LanczosResult(energy=-2.5, eigenvector=v)
    assert len(lr) == 2
    assert lr[0] == -2.5
    assert np.array_equal(lr[1], v)
    e, vec = lr
    assert e == -2.5
    assert np.array_equal(vec, v)

    # DavidsonResult protocol
    dr = DavidsonResult(eigenvalues=np.array([1.0, 2.0]), eigenvectors=[v, v])
    assert len(dr) == 2
    assert dr[0][0] == 1.0
    evs, evecs = dr
    assert len(evs) == 2
    assert len(evecs) == 2

    # DynamicsResult protocol
    dyn = DynamicsResult(alphas=[1.0, 2.0], betas=[0.5], norm_phi0=1.0)
    assert len(dyn) == 3
    a, b, norm = dyn
    assert len(a) == 2
    assert len(b) == 1
    assert norm == 1.0

    # FTLMResult protocol
    class DummyCppFTLM:
        beta = 1.5
        partition_function = 10.0
        internal_energy = -3.2
        specific_heat = 0.8
    ft = FTLMResult(DummyCppFTLM())
    assert len(ft) == 4
    beta, Z, E, C = ft
    assert beta == 1.5
    assert Z == 10.0
    assert E == -3.2
    assert C == 0.8


def test_hamiltonian_2arg_inference():
    # Test qk.Hamiltonian(basis, ops) where site is omitted
    basis = qk.basis.SpinHalf(4, sz=0)
    os = qk.OpSum()
    for i in range(3):
        os += 1.0 * qk.Sz(i) * qk.Sz(i + 1)
    H = qk.Hamiltonian(basis, os)
    assert H.dimension == 6
    assert isinstance(H.site, qk.site.SpinHalfSite)

    # Matmul via @
    x = np.ones(6, dtype=np.complex128)
    y = H @ x
    assert len(y) == 6

    # Device placement
    H_cpu = H.to(device="cpu")
    assert H_cpu.dimension == 6
    y_cpu = H_cpu @ x
    assert np.allclose(y, y_cpu)


def test_davidson_oop_solver():
    _, _, _, H = _build_heisenberg_model(N=4, dtype=np.float64)
    solver = qk.solvers.Davidson(n_eig=1, max_subspace=10, tol=1e-8)
    res = solver.solve(H)
    assert isinstance(res, DavidsonResult)
    evals, evecs = solver.solve(H)
    assert math.isclose(evals[0], -1.6160254038, abs_tol=1e-5)
    assert len(evecs) == 1
    assert len(evecs[0]) == 6


def test_continued_fraction_oop_solver():
    _, _, _, H = _build_heisenberg_model(N=4, dtype=np.float64)
    phi0 = np.zeros(H.dimension, dtype=np.complex128)
    phi0[0] = 1.0
    solver = qk.solvers.ContinuedFraction(n_iter=10)
    res = solver.solve(H, phi0)
    assert isinstance(res, DynamicsResult)
    alphas, betas, norm = solver.solve(H, phi0)
    assert len(alphas) > 0
    assert len(betas) == len(alphas) - 1
    assert math.isclose(norm, 1.0, abs_tol=1e-5)


def test_correction_vector_oop_solver():
    _, _, _, H = _build_heisenberg_model(N=4, dtype=np.float64)
    # Find ground state
    e0, psi0 = qk.solvers.Lanczos(maxiter=100, tol=1e-10).solve(H)
    # Apply excitation
    ops = qk.OpSum()
    ops += 1.0 * qk.Sz(0)
    H_sz = qk.Hamiltonian(H.basis, ops)
    op_psi0 = H_sz @ psi0

    solver = qk.solvers.CorrectionVector(e0=e0, omega=1.0, eta=0.1, max_iter=200, tol=1e-6)
    res = solver.solve(H, op_psi0)
    assert isinstance(res, CorrectionVectorResult)
    assert res.spectral_function >= 0.0
    assert len(res.correction_vector) == H.dimension
    corr_vec, spec, iters, conv = solver.solve(H, op_psi0)
    assert math.isclose(spec, res.spectral_function, abs_tol=1e-12)


def test_ftlm_oop_solver():
    _, _, _, H = _build_heisenberg_model(N=4, dtype=np.float64)
    solver = FTLM(beta=1.0, n_random=10, n_steps=20, seed=42)
    res = solver.solve(H)
    assert isinstance(res, FTLMResult)
    assert math.isclose(res.beta, 1.0, abs_tol=1e-12)
    assert res.partition_function > 0.0

    # Test tuple unpacking
    b, z, e, cv = solver.solve(H)
    assert math.isclose(b, 1.0, abs_tol=1e-12)
    assert z > 0.0


def test_ftlm_sweep_oop_solver():
    _, _, _, H = _build_heisenberg_model(N=4, dtype=np.float64)
    betas = [0.2, 0.5, 1.0, 2.0]
    solver = FTLM(n_random=15, n_steps=25, seed=123)
    res = solver.solve(H, betas=betas, observables=[H])

    assert isinstance(res, FTLMSweepResult)
    assert len(res.beta_grid) == len(betas)
    assert np.allclose(res.beta_grid, betas)
    assert len(res.partition_functions) == len(betas)
    assert np.all(res.partition_functions > 0.0)
    assert len(res.internal_energies) == len(betas)
    assert len(res.free_energies) == len(betas)
    assert len(res.specific_heats) == len(betas)
    assert np.all(res.specific_heats >= -1e-12)
    assert len(res.entropies) == len(betas)
    assert np.all(res.entropies >= -1e-12)

    # Check observable expectations for H match internal energy algebraically
    assert len(res.observable_expectations) == 1
    obs_H = res.observable_expectations[0]
    assert len(obs_H) == len(betas)
    assert np.allclose(obs_H, res.internal_energies, atol=1e-5, rtol=1e-4)

    # Check error bars array exists and has same shape
    assert len(res.observable_errors) == 1
    assert len(res.observable_errors[0]) == len(betas)
    assert np.all(res.observable_errors[0] >= 0.0)

    # Also test functional convenience API
    res_fn = qk.solvers.ftlm(H, betas=betas, observables=[H], n_random=15, n_steps=25, seed=123)
    assert isinstance(res_fn, FTLMSweepResult)
    assert np.allclose(res_fn.internal_energies, res.internal_energies)


def test_ftlm_decoupled_workflow():
    _, _, _, H = _build_heisenberg_model(N=4, dtype=np.float64)
    betas = [0.2, 0.5, 1.0, 2.0]
    solver = FTLM(n_random=15, n_steps=25, seed=123)

    # Reference sweep
    ref_sweep = solver.solve(H, betas=betas, observables=[H])

    # Stage 1: Sample once
    samples = solver.sample(H, observables=[H])
    assert isinstance(samples, qk.solvers.FTLMSamples)
    assert len(samples) == 15
    assert samples.num_samples == 15

    # Stage 2: Evaluate sweep
    res1 = solver.evaluate_sweep(samples, betas=betas)
    assert isinstance(res1, FTLMSweepResult)
    assert np.allclose(res1.partition_functions, ref_sweep.partition_functions)
    assert np.allclose(res1.internal_energies, ref_sweep.internal_energies)

    # Stage 2 via sample method
    res1_method = samples.evaluate_sweep(betas)
    assert np.allclose(res1_method.internal_energies, ref_sweep.internal_energies)

    # Zero-cost re-evaluation on new grid
    dense_betas = np.linspace(0.1, 5.0, 25)
    res_dense = samples.evaluate_sweep(dense_betas)
    assert isinstance(res_dense, FTLMSweepResult)
    assert len(res_dense.beta_grid) == 25
    assert np.all(res_dense.partition_functions > 0.0)

    # Functional API: ftlm_sample and ftlm_evaluate_sweep
    samples_fn = qk.solvers.ftlm_sample(H, observables=[H], n_random=15, n_steps=25, seed=123)
    res_fn = qk.solvers.ftlm_evaluate_sweep(samples_fn, betas)
    assert np.allclose(res_fn.internal_energies, ref_sweep.internal_energies)

    # Test FP32 decoupled FTLM
    _, _, _, H32 = _build_heisenberg_model(N=4, dtype=np.float32)
    solver32 = FTLM(n_random=10, n_steps=20, seed=42)
    samples32 = solver32.sample(H32, observables=[H32])
    assert isinstance(samples32, qk.solvers.FTLMSamples)
    res32 = samples32.evaluate_sweep([0.5, 1.0, 2.0])
    assert isinstance(res32, FTLMSweepResult)
    assert len(res32.beta_grid) == 3


def test_ftlm_streamed_workflow():
    _, _, _, H = _build_heisenberg_model(N=4, dtype=np.float64)
    betas = [0.2, 0.5, 1.0, 2.0]
    
    # Streamed mode (memory-bounded two-pass)
    res_streamed = qk.solvers.ftlm_sweep_streamed(
        H, betas=betas, observables=[H], n_random=20, n_steps=25, seed=123
    )
    assert isinstance(res_streamed, FTLMSweepResult)
    assert res_streamed.dimension == 6
    assert len(res_streamed.effective_samples) == len(betas)
    assert np.all(res_streamed.effective_samples > 0.0)

    # Cached mode with same seed
    res_cached = qk.solvers.ftlm(
        H, betas=betas, observables=[H], n_random=20, n_steps=25, seed=123, mode="cached"
    )
    assert isinstance(res_cached, FTLMSweepResult)
    assert res_cached.dimension == 6

    # Both modes should match closely
    assert np.allclose(res_streamed.partition_functions, res_cached.partition_functions, rtol=1e-5)
    assert np.allclose(res_streamed.internal_energies, res_cached.internal_energies, atol=1e-4)


def test_real_time_dynamics():
    # 2-site Heisenberg model: H = Sz0*Sz1 + 0.5*(Sp0*Sm1 + Sm0*Sp1)
    basis = qk.SpinHalfBasis(N=2, dtype=np.float64)
    site = qk.SpinHalfSite(dtype=np.float64)
    opsum = qk.OpSum()
    opsum += (1.0, "Sz", 0, "Sz", 1)
    opsum += (0.5, "Sp", 0, "Sm", 1)
    opsum += (0.5, "Sm", 0, "Sp", 1)
    H = qk.MatrixFreeHamiltonian(basis, site, opsum, dtype=np.float64)

    # Observable Sz0
    op_sz0 = qk.OpSum()
    op_sz0 += (1.0, "Sz", 0)
    Sz0 = qk.MatrixFreeHamiltonian(basis, site, op_sz0, dtype=np.float64)

    # Observable Sz1
    op_sz1 = qk.OpSum()
    op_sz1 += (1.0, "Sz", 1)
    Sz1 = qk.MatrixFreeHamiltonian(basis, site, op_sz1, dtype=np.float64)

    # Find |up, down> state
    d0 = Sz0.diagonal()
    d1 = Sz1.diagonal()
    psi0 = np.zeros(H.dimension, dtype=np.complex128)
    for i in range(H.dimension):
        if math.isclose(d0[i].real, 0.5, abs_tol=1e-5) and math.isclose(d1[i].real, -0.5, abs_tol=1e-5):
            psi0[i] = 1.0
            break

    time_grid = [0.0, 0.5, 1.0, 1.5, 2.0, math.pi]
    res = qk.solvers.time_evolve(H, psi0, time_grid, observables=[Sz0], n_steps=10)
    assert isinstance(res, qk.solvers.RealTimeResult)
    assert len(res.time_grid) == len(time_grid)
    assert len(res.survival_probabilities) == len(time_grid)
    assert len(res.observable_expectations) == 1

    # Check Rabi oscillations: <Sz0>(t) = 0.5 * cos(Delta E * t) with Delta E = 1.0
    for t, exp_val in zip(time_grid, res.observable_expectations[0]):
        expected = 0.5 * math.cos(t)
        assert math.isclose(exp_val.real, expected, abs_tol=1e-3)
        assert math.isclose(exp_val.imag, 0.0, abs_tol=1e-3)


def test_ftlm_dynamics():
    basis = qk.SpinHalfBasis(N=2, dtype=np.float64)
    site = qk.SpinHalfSite(dtype=np.float64)
    opsum = qk.OpSum()
    opsum += (1.0, "Sz", 0, "Sz", 1)
    opsum += (0.5, "Sp", 0, "Sm", 1)
    opsum += (0.5, "Sm", 0, "Sp", 1)
    H = qk.MatrixFreeHamiltonian(basis, site, opsum, dtype=np.float64)

    op_sz0 = qk.OpSum()
    op_sz0 += (1.0, "Sz", 0)
    Sz0 = qk.MatrixFreeHamiltonian(basis, site, op_sz0, dtype=np.float64)

    time_grid = [0.0, 0.5, 1.0]
    res = qk.solvers.ftlm_dynamics(H, Sz0, Sz0, beta=1.0, time_grid=time_grid, n_random=50, n_steps=10, seed=42)
    assert isinstance(res, qk.solvers.FTLMDynamicsResult)
    assert len(res.time_grid) == 3
    assert len(res.correlations) == 3
    assert len(res.correlation_errors) == 3

    # At t=0, C_AB(0) = <Sz0^2> = 1/4 = 0.25
    assert math.isclose(res.correlations[0].real, 0.25, abs_tol=0.05)
    assert math.isclose(res.correlations[0].imag, 0.0, abs_tol=0.05)




