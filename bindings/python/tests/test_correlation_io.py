import qkrylov as qk
import numpy as np
import pytest
import math

def test_save_load_correlation_python(tmp_path):
    N = 4
    basis = qk.SpinHalfBasis(N, sz=0)
    site = qk.SpinHalfSite()

    # 1. Solve 4-site Heisenberg model ground state
    os_h = qk.OpSum()
    for i in range(N):
        j = (i + 1) % N
        os_h += 1.0, "Sz", i, "Sz", j
        os_h += 0.5, "Sp", i, "Sm", j
        os_h += 0.5, "Sm", i, "Sp", j

    H = qk.MatrixFreeHamiltonian(basis, site, os_h)
    l_res = qk.lanczos_ground_state(H, maxiter=100, tol=1e-12)

    # 2. Save ground state result to HDF5
    h5_file = tmp_path / "ground_state.h5"
    l_res.save(h5_file)

    # 3. Load ground state result back for reuse in future calculation
    loaded_res = qk.LanczosResult.load(h5_file)

    assert math.isclose(loaded_res.energy, l_res.energy, abs_tol=1e-12)
    assert np.allclose(loaded_res.eigenvector, l_res.eigenvector)

    # 4. Calculate spin-spin correlation <psi_0 | Sz_0 Sz_2 | psi_0> using loaded eigenvector
    os_corr = qk.OpSum()
    os_corr += 1.0, "Sz", 0, "Sz", 2
    Sz0Sz2 = qk.MatrixFreeHamiltonian(basis, site, os_corr)

    A_psi0 = Sz0Sz2.apply(loaded_res.eigenvector)
    corr_val = np.vdot(loaded_res.eigenvector, A_psi0)

    print("<Sz_0 Sz_2> correlation =", corr_val)
    assert math.isclose(corr_val.imag, 0.0, abs_tol=1e-10)
    assert corr_val.real > 0.0
