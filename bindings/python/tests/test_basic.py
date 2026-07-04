import qkrylov
import pytest

def test_basis():
    N = 4
    basis = qkrylov.SpinHalfBasis(N)
    assert basis.size() == 16
    assert basis.nsites() == 4

def test_fermion_basis():
    sec = qkrylov.Sector()
    sec.use_n = True
    sec.n = 2
    basis = qkrylov.FermionBasis(4, sec)
    assert basis.size() == 6

def test_hamiltonian():
    N = 2
    basis = qkrylov.SpinHalfBasis(N)
    site = qkrylov.SpinHalfSite()
    os = qkrylov.OpSum()

    term = qkrylov.OperatorTerm()
    term.coeff = 1.0
    term.factors = [qkrylov.OperatorFactor("Sz", 0), qkrylov.OperatorFactor("Sz", 1)]
    os.add_term(term)

    H = qkrylov.MatrixFreeHamiltonian(basis, site, os)
    assert H.dimension() == 4
    diag = H.diagonal()
    assert len(diag) == 4
