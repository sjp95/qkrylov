"""Python interface for qkrylov.

A modern C++20 framework for matrix-free Krylov methods in quantum many-body physics,
now with a Pythonic wrapper layer.
"""

from . import _qkrylov_cpp as _cpp
Sector = _cpp.Sector

from .operators import (
    Op, OpSum,
    Sz, Sp, Sm, Sx, Sy,
    CdagUp, CUp, CdagDn, CDn,
    Nup, Ndn, Nupdn,
    Bdag, B, N
)
from .site import Site, SpinHalfSite, SpinSSite, FermionSite, HubbardSite, TJSite
from .basis import Basis, SpinHalfBasis, FermionBasis, HubbardBasis, TJBasis, SpinSBasis
from .hamiltonian import MatrixFreeHamiltonian
from .solvers import (
    LanczosResult,
    lanczos_ground_state,
    DavidsonResult,
    davidson_lowest,
    DynamicsResult,
    continued_fraction_coeffs,
    evaluate_spectral_function,
    CorrectionVectorResult,
    correction_vector_spectral,
    FTLMResult,
    ftlm,
)

try:
    from importlib.metadata import version as _metadata_version
    __version__ = _metadata_version("qkrylov")
except Exception:
    __version__ = "0.0.0"

__all__ = [
    # Sector
    "Sector",

    # Operators
    "Op",
    "OpSum",
    
    # Operator Generators
    "Sz", "Sp", "Sm", "Sx", "Sy",
    "CdagUp", "CUp", "CdagDn", "CDn",
    "Nup", "Ndn", "Nupdn",
    "Bdag", "B", "N",
    
    # Sites
    "Site",
    "SpinHalfSite",
    "SpinSSite",
    "FermionSite",
    "HubbardSite",
    "TJSite",
    
    # Bases
    "Basis",
    "SpinHalfBasis",
    "FermionBasis",
    "HubbardBasis",
    "TJBasis",
    "SpinSBasis",
    
    # Hamiltonian
    "MatrixFreeHamiltonian",
    
    # Solvers
    "LanczosResult",
    "lanczos_ground_state",
    "DavidsonResult",
    "davidson_lowest",
    "DynamicsResult",
    "continued_fraction_coeffs",
    "evaluate_spectral_function",
    "CorrectionVectorResult",
    "correction_vector_spectral",
    "FTLMResult",
    "ftlm",
]
