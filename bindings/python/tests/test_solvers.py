import qkrylov as qk
import numpy as np
import pytest
import math

def test_lanczos_solver():
    N = 4
    basis = qk.SpinHalfBasis(N, sz=0)
    site = qk.SpinHalfSite()
    os = qk.OpSum()

    for i in range(N):
        j = (i + 1) % N
        os += 1.0, "Sz", i, "Sz", j
        os += 0.5, "Sp", i, "Sm", j
        os += 0.5, "Sm", i, "Sp", j

    H = qk.MatrixFreeHamiltonian(basis, site, os)
    res = qk.lanczos_ground_state(H, maxiter=100, tol=1e-12)

    assert math.isclose(res.energy, -2.0, abs_tol=1e-6)
    assert len(res.eigenvector) == H.dimension
    assert isinstance(res.eigenvector, np.ndarray)

def test_davidson_solver():
    N = 4
    basis = qk.SpinHalfBasis(N, sz=0)
    site = qk.SpinHalfSite()
    os = qk.OpSum()

    for i in range(N):
        j = (i + 1) % N
        os += 1.0, "Sz", i, "Sz", j
        os += 0.5, "Sp", i, "Sm", j
        os += 0.5, "Sm", i, "Sp", j

    H = qk.MatrixFreeHamiltonian(basis, site, os)

    # Solve for lowest 2 eigenvalues
    d_res = qk.davidson_lowest(H, n_eig=2, max_subspace=20, tol=1e-8)

    assert len(d_res.eigenvalues) == 2
    assert len(d_res.eigenvectors) == 2
    assert math.isclose(d_res.eigenvalues[0], -2.0, abs_tol=1e-6)
    assert d_res.eigenvalues[1] >= d_res.eigenvalues[0]

def test_continued_fraction_dynamics():
    N = 4
    basis = qk.SpinHalfBasis(N, sz=0)
    site = qk.SpinHalfSite()
    os_h = qk.OpSum()

    for i in range(N):
        j = (i + 1) % N
        os_h += 1.0, "Sz", i, "Sz", j
        os_h += 0.5, "Sp", i, "Sm", j
        os_h += 0.5, "Sm", i, "Sp", j

    H = qk.MatrixFreeHamiltonian(basis, site, os_h)
    l_res = qk.lanczos_ground_state(H, maxiter=100, tol=1e-12)
    E0 = l_res.energy

    # Prepare excitation phi0 = Sz(0) |psi0>
    os_sz0 = qk.OpSum()
    os_sz0 += 1.0, "Sz", 0
    Sz0 = qk.MatrixFreeHamiltonian(basis, site, os_sz0)

    phi0 = Sz0.apply(l_res.eigenvector)
    dyn_res = qk.continued_fraction_coeffs(H, phi0, n_iter=20)

    assert len(dyn_res.alphas) > 0
    assert dyn_res.norm_phi0 > 0.0

    S_w = qk.evaluate_spectral_function(dyn_res, omega=1.0, E0=E0, eta=0.1)
    assert S_w >= 0.0

def test_correction_vector_spectral():
    N = 4
    basis = qk.SpinHalfBasis(N, sz=0)
    site = qk.SpinHalfSite()
    os_h = qk.OpSum()

    for i in range(N):
        j = (i + 1) % N
        os_h += 1.0, "Sz", i, "Sz", j
        os_h += 0.5, "Sp", i, "Sm", j
        os_h += 0.5, "Sm", i, "Sp", j

    H = qk.MatrixFreeHamiltonian(basis, site, os_h)
    l_res = qk.lanczos_ground_state(H, maxiter=100, tol=1e-12)
    E0 = l_res.energy

    os_sz0 = qk.OpSum()
    os_sz0 += 1.0, "Sz", 0
    Sz0 = qk.MatrixFreeHamiltonian(basis, site, os_sz0)

    Op_psi0 = Sz0.apply(l_res.eigenvector)
    cv_res = qk.correction_vector_spectral(H, Op_psi0, E0=E0, omega=1.0, eta=0.1)

    assert cv_res.spectral_function >= 0.0
    assert len(cv_res.correction_vector) == H.dimension
