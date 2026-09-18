import pytest
import numpy as np
import math

import qkrylov as qk
from qkrylov import (
    SpinSSite,
    SpinHalfSite,
    SpinSBasis,
    SpinHalfBasis,
    MatrixFreeHamiltonian,
    OpSum,
    Sz,
    Sp,
    Sm,
    lanczos_ground_state,
    correction_vector,
    correction_vector_spectral,
    CorrectionVectorResult,
)


# ==============================================================================
# Tier 1 & Tier 2: SpinSSite Unit & Boundary Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("S, expected_dim", [
    (0.5, 2),
    (1.0, 3),
    (1.5, 4),
    (2.0, 5),
    (2.5, 6),
])
def test_spin_s_site_valid_parameters(S, expected_dim, dtype):
    """Test SpinSSite construction, properties, and dimensions for standard spins."""
    site = SpinSSite(S=S, dtype=dtype)
    assert math.isclose(site.spin, S, abs_tol=1e-6)
    assert site.dimension_per_site == expected_dim
    assert "SpinSSite" in repr(site)
    assert str(S) in repr(site)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("invalid_S", [0.0, -0.5, -1.0, -2.0])
def test_spin_s_site_invalid_spin_raises(invalid_S, dtype):
    """Test SpinSSite rejects non-positive spin quantum numbers."""
    with pytest.raises(RuntimeError, match="Spin S must be positive"):
        SpinSSite(S=invalid_S, dtype=dtype)


def test_spin_s_site_default_arguments():
    """Test SpinSSite default spin is 0.5."""
    site = SpinSSite()
    assert math.isclose(site.spin, 0.5, abs_tol=1e-6)
    assert site.dimension_per_site == 2


# ==============================================================================
# Tier 1 & Tier 2: SpinSBasis Unit, Boundary & Equivalence Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("N, S, expected_dim", [
    (1, 0.5, 2),
    (2, 0.5, 4),
    (3, 0.5, 8),
    (4, 0.5, 16),
    (1, 1.0, 3),
    (2, 1.0, 9),
    (3, 1.0, 27),
    (4, 1.0, 81),
    (1, 1.5, 4),
    (2, 1.5, 16),
    (1, 2.0, 5),
    (2, 2.0, 25),
])
def test_spin_s_basis_full_dimensions(N, S, expected_dim, dtype):
    """Test unconstrained SpinSBasis dimension equals (2S+1)^N."""
    basis = SpinSBasis(N=N, S=S, conserve_sz=False, dtype=dtype)
    assert basis.nsites == N
    assert math.isclose(basis.spin, S, abs_tol=1e-6)
    expected_local_d = int(round(2.0 * S + 1.0))
    assert basis.dimension_per_site == expected_local_d
    assert basis.size == expected_dim
    assert "SpinSBasis" in repr(basis)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("N, S, sz, expected_dim", [
    # Spin-1 (d=3): N=2, total Sz in {-2, -1, 0, 1, 2}
    (2, 1.0,  0.0, 3),   # (-1,1), (0,0), (1,-1)
    (2, 1.0,  1.0, 2),   # (0,1), (1,0)
    (2, 1.0, -1.0, 2),   # (0,-1), (-1,0)
    (2, 1.0,  2.0, 1),   # (1,1)
    (2, 1.0, -2.0, 1),   # (-1,-1)
    # Spin-1: N=4, Sz=0 has dim 19
    (4, 1.0,  0.0, 19),
    # Spin-1/2 (d=2): N=4
    (4, 0.5,  0.0, 6),   # Binomial(4, 2) = 6
    (4, 0.5,  1.0, 4),   # Binomial(4, 3) = 4
    (4, 0.5, -1.0, 4),   # Binomial(4, 1) = 4
    (4, 0.5,  2.0, 1),   # Binomial(4, 4) = 1
    (4, 0.5, -2.0, 1),   # Binomial(4, 0) = 1
])
def test_spin_s_basis_sz_sectors(N, S, sz, expected_dim, dtype):
    """Test Sz-conserved SpinSBasis sector sizes."""
    basis = SpinSBasis(N=N, S=S, sz=sz, dtype=dtype)
    assert basis.nsites == N
    assert basis.size == expected_dim
    assert "sz=" in repr(basis)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("N", [2, 4, 6])
def test_spin_s_basis_equivalence_with_spinhalf_full(N, dtype):
    """Test SpinSBasis(S=0.5) dimensions match SpinHalfBasis exactly for full space."""
    s_basis = SpinSBasis(N=N, S=0.5, conserve_sz=False, dtype=dtype)
    sh_basis = SpinHalfBasis(N=N, conserve_sz=False, dtype=dtype)
    assert s_basis.size == sh_basis.size
    assert s_basis.nsites == sh_basis.nsites


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("N, sz", [
    (2, 0.0),
    (2, 1.0),
    (2, -1.0),
    (4, 0.0),
    (4, 1.0),
    (4, -1.0),
    (4, 2.0),
])
def test_spin_s_basis_equivalence_with_spinhalf_sz(N, sz, dtype):
    """Test SpinSBasis(S=0.5, sz) sector dimensions match SpinHalfBasis(N, sz)."""
    s_basis = SpinSBasis(N=N, S=0.5, sz=sz, dtype=dtype)
    sh_basis = SpinHalfBasis(N=N, sz=sz, dtype=dtype)
    assert s_basis.size == sh_basis.size
    assert s_basis.nsites == sh_basis.nsites


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_spin_s_basis_state_index_contains_methods(dtype):
    """Test state(), index(), and contains() methods and consistency."""
    basis = SpinSBasis(N=4, S=1.0, sz=0, dtype=dtype)
    assert basis.size == 19

    states = []
    for i in range(basis.size):
        s = basis.state(i)
        states.append(s)
        assert basis.contains(s) is True
        assert basis.index(s) == i

    # Verify states are strictly sorted
    assert states == sorted(states)

    # Verify out-of-sector / invalid state
    # Max valid state in 4-site spin-1 full basis is 3^4 - 1 = 80
    invalid_state = 10000
    assert basis.contains(invalid_state) is False
    with pytest.raises(RuntimeError, match="State not present in basis"):
        basis.index(invalid_state)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("invalid_N", [0, -1, -5])
def test_spin_s_basis_invalid_nsites_raises(invalid_N, dtype):
    """Test SpinSBasis rejects non-positive site count."""
    with pytest.raises(RuntimeError, match="N must be > 0"):
        SpinSBasis(N=invalid_N, S=1.0, dtype=dtype)


# ==============================================================================
# Tier 3 & Tier 4: Correction Vector Solver E2E Physics Tests
# ==============================================================================

def _build_heisenberg_chain(N, S, sz, dtype):
    """Helper to build a 1D Heisenberg chain with SpinSSite and SpinSBasis."""
    basis = SpinSBasis(N=N, S=S, sz=sz, dtype=dtype)
    site = SpinSSite(S=S, dtype=dtype)
    ops = OpSum(dtype=dtype)
    for i in range(N - 1):
        ops += 1.0 * Sz(i) * Sz(i + 1) + 0.5 * (Sp(i) * Sm(i + 1) + Sm(i) * Sp(i + 1))
    H = MatrixFreeHamiltonian(basis, site, ops, dtype=dtype)
    return basis, site, H


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_spin1_correction_vector_4site_heisenberg_e2e(dtype):
    """E2E test: 4-site Spin-1 Heisenberg chain correction vector spectroscopy."""
    N = 4
    S = 1.0
    sz = 0.0

    basis, site, H = _build_heisenberg_chain(N, S, sz, dtype)
    assert H.dimension == 19

    # Solve ground state using Lanczos
    tol_lanczos = 1e-6 if dtype == np.float32 else 1e-12
    lgs = lanczos_ground_state(H, maxiter=200, tol=tol_lanczos)
    assert len(lgs.eigenvector) == 19
    # Ground state energy for 4-site Spin-1 chain is approx -4.64575
    assert lgs.energy < -4.0

    # Apply excitation operator Sz(0) to ground state
    ops_sz0 = OpSum(dtype=dtype)
    ops_sz0 += 1.0 * Sz(0)
    H_sz0 = MatrixFreeHamiltonian(basis, site, ops_sz0, dtype=dtype)
    op_psi0 = H_sz0.apply(lgs.eigenvector)
    assert len(op_psi0) == 19

    # Correction vector solver at omega = 1.5, eta = 0.1
    omega = 1.5
    eta = 0.1
    tol_cg = 1e-5 if dtype == np.float32 else 1e-8
    res = correction_vector(
        H=H,
        op_psi0=op_psi0,
        E0=lgs.energy,
        omega=omega,
        eta=eta,
        max_iter=500,
        tol=tol_cg
    )

    assert isinstance(res, CorrectionVectorResult)
    assert res.converged is True
    assert res.iterations > 0
    assert res.iterations <= 500
    assert res.spectral_function >= 0.0
    assert res.correction_vector.shape == (19,)

    # Spectral function S(omega=1.5) should be positive and physically sensible
    assert res.spectral_function > 1e-4

    # Verify unpackability of CorrectionVectorResult
    corr_vec, spec_val, iters, conv = res
    assert np.array_equal(corr_vec, res.correction_vector)
    assert spec_val == res.spectral_function
    assert iters == res.iterations
    assert conv == res.converged

    # Verify repr string
    assert "CorrectionVectorResult" in repr(res)
    assert "spectral_function" in repr(res)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_correction_vector_zero_vector_edge_case(dtype):
    """Edge case: Passing zero vector to correction_vector solver."""
    N = 2
    S = 1.0
    basis, site, H = _build_heisenberg_chain(N, S, sz=0.0, dtype=dtype)

    zero_vec = np.zeros(H.dimension, dtype=np.complex128 if dtype == np.float64 else np.complex64)
    res = correction_vector(
        H=H,
        op_psi0=zero_vec,
        E0=-2.0,
        omega=1.0,
        eta=0.1
    )

    assert res.converged is True
    assert res.iterations == 0
    assert res.spectral_function == 0.0
    assert np.all(res.correction_vector == 0.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_correction_vector_alias_equivalence(dtype):
    """Test correction_vector is identical to correction_vector_spectral."""
    assert correction_vector is correction_vector_spectral

    N = 2
    S = 1.0
    basis, site, H = _build_heisenberg_chain(N, S, sz=0.0, dtype=dtype)
    vec = np.ones(H.dimension, dtype=np.complex128 if dtype == np.float64 else np.complex64)

    res1 = correction_vector(H, vec, E0=-1.0, omega=0.5, eta=0.1)
    res2 = correction_vector_spectral(H, vec, E0=-1.0, omega=0.5, eta=0.1)

    assert res1.converged == res2.converged
    assert res1.iterations == res2.iterations
    assert math.isclose(res1.spectral_function, res2.spectral_function, rel_tol=1e-5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("omega", [0.5, 1.0, 1.5, 2.0, 3.0])
def test_correction_vector_frequency_scan(omega, dtype):
    """Multi-frequency scan: S(omega) must remain non-negative across frequencies."""
    N = 2
    S = 1.0
    basis, site, H = _build_heisenberg_chain(N, S, sz=0.0, dtype=dtype)

    lgs = lanczos_ground_state(H)
    ops_sz0 = OpSum(dtype=dtype)
    ops_sz0 += 1.0 * Sz(0)
    H_sz0 = MatrixFreeHamiltonian(basis, site, ops_sz0, dtype=dtype)
    op_psi0 = H_sz0.apply(lgs.eigenvector)

    tol = 1e-5 if dtype == np.float32 else 1e-8
    res = correction_vector(H, op_psi0, E0=lgs.energy, omega=omega, eta=0.1, tol=tol)
    assert res.converged is True
    assert res.spectral_function >= 0.0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("eta", [0.05, 0.1, 0.2])
def test_correction_vector_broadening_parameter(eta, dtype):
    """Test varying broadening factor eta."""
    N = 2
    S = 1.0
    basis, site, H = _build_heisenberg_chain(N, S, sz=0.0, dtype=dtype)

    lgs = lanczos_ground_state(H)
    ops_sz0 = OpSum(dtype=dtype)
    ops_sz0 += 1.0 * Sz(0)
    H_sz0 = MatrixFreeHamiltonian(basis, site, ops_sz0, dtype=dtype)
    op_psi0 = H_sz0.apply(lgs.eigenvector)

    tol = 1e-5 if dtype == np.float32 else 1e-8
    res = correction_vector(H, op_psi0, E0=lgs.energy, omega=1.0, eta=eta, tol=tol)
    assert res.converged is True
    assert res.spectral_function >= 0.0


def test_correction_vector_wrong_dimension_raises():
    """Safety: Passing a state vector with incorrect dimension raises error."""
    N = 2
    S = 1.0
    basis, site, H = _build_heisenberg_chain(N, S, sz=0.0, dtype=np.float32)

    # H.dimension is 3, pass 10 elements
    wrong_vec = np.zeros(10, dtype=np.complex64)
    with pytest.raises(ValueError):
        correction_vector(H, wrong_vec, E0=-1.0, omega=1.0)
