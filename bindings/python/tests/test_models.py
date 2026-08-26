import qkrylov as qk
import numpy as np
import pytest
import math

def test_dmi_and_complex_operators():
    # Test DMI interaction: D * (S_i x S_j)_z = (i D / 2) Sp_i Sm_j - (i D / 2) Sm_i Sp_j
    N = 3
    basis = qk.SpinHalfBasis(N)
    site = qk.SpinHalfSite()

    J = 1.0
    D = 0.5
    os = qk.OpSum()

    for i in range(N - 1):
        j = i + 1
        os += J, "Sz", i, "Sz", j
        os += 0.5 * J, "Sp", i, "Sm", j
        os += 0.5 * J, "Sm", i, "Sp", j
        # Complex coefficients
        os += 0.5j * D, "Sp", i, "Sm", j
        os += -0.5j * D, "Sm", i, "Sp", j

    H = qk.MatrixFreeHamiltonian(basis, site, os)
    res = qk.lanczos_ground_state(H)
    assert res.energy < 0.0

def test_spin_s_model():
    basis = qk.SpinSBasis(2, S=1.0)
    site = qk.SpinSSite(1.0)
    assert basis.size == 9
    assert site._cpp_obj.spin() == 1.0

    os = qk.OpSum()
    os += 1.0, "Sz", 0, "Sz", 1
    os += 0.5, "Sp", 0, "Sm", 1
    os += 0.5, "Sm", 0, "Sp", 1

    H = qk.MatrixFreeHamiltonian(basis, site, os)
    res = qk.lanczos_ground_state(H)
    assert math.isclose(res.energy, -2.0, abs_tol=1e-5)

def test_hubbard_model():
    # 2-site Hubbard model with Nup=1, Ndn=1
    sec = qk.Sector()
    sec.use_nup = True
    sec.use_ndn = True
    sec.nup = 1
    sec.ndn = 1

    basis = qk.HubbardBasis(2, sector=sec)
    site = qk.HubbardSite()

    assert basis.size == 4

    t = 1.0
    U = 4.0
    os = qk.OpSum()
    os += -t, "CdagUp", 0, "CUp", 1
    os += -t, "CdagUp", 1, "CUp", 0
    os += -t, "CdagDn", 0, "CDn", 1
    os += -t, "CdagDn", 1, "CDn", 0
    os += U, "Nup", 0, "Ndn", 0
    os += U, "Nup", 1, "Ndn", 1

    H = qk.MatrixFreeHamiltonian(basis, site, os)
    res = qk.lanczos_ground_state(H)

    exact_e = (U - math.sqrt(U**2 + 16.0 * t**2)) / 2.0
    assert math.isclose(res.energy, exact_e, abs_tol=1e-5)

def test_tj_model():
    sec = qk.Sector()
    sec.use_nup = True
    sec.use_ndn = True
    sec.nup = 1
    sec.ndn = 0

    basis = qk.TJBasis(2, sector=sec)
    site = qk.TJSite()

    assert basis.size == 2

    os = qk.OpSum()
    os += -1.0, "CdagUp", 0, "CUp", 1
    os += -1.0, "CdagUp", 1, "CUp", 0

    H = qk.MatrixFreeHamiltonian(basis, site, os)
    res = qk.lanczos_ground_state(H)
    assert math.isclose(res.energy, -1.0, abs_tol=1e-6)

def test_spinless_fermion_model():
    sec = qk.Sector()
    sec.use_n = True
    sec.n = 2

    basis = qk.FermionBasis(4, sector=sec)
    site = qk.FermionSite()

    assert basis.size == 6

    os = qk.OpSum()
    t = 1.0
    V = 2.0
    for i in range(4):
        j = (i + 1) % 4
        os += -t, "Cdag", i, "C", j
        os += -t, "Cdag", j, "C", i
        os += V, "N", i, "N", j

    H = qk.MatrixFreeHamiltonian(basis, site, os)
    res = qk.lanczos_ground_state(H)
    assert isinstance(res.energy, float)
