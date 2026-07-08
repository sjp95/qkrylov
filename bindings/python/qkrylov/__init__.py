"""Python interface for qkrylov."""

from .qkrylov_cpp import (
    Sector,
    OperatorFactor,
    OperatorTerm,
    OpSum,
    Basis,
    SpinHalfBasis,
    FermionBasis,
    HubbardBasis,
    TJBasis,
    GenericBasis,
    Site,
    SpinHalfSite,
    FermionSite,
    HubbardSite,
    TJSite,
    BosonSite,
    HardcoreBosonSite,
    SpinSSite,
    MatrixFreeHamiltonian,
)

try:
    from .qkrylov_cpp import CUDAHamiltonian
except ImportError:
    CUDAHamiltonian = None

from .qkrylov_cpp import (
    LanczosResult,
    lanczos_ground_state,
    DavidsonResult,
    davidson_lowest,
    DynamicsResult,
    continued_fraction_coeffs,
    evaluate_spectral_function,
    FTLMResult,
    ftlm,
)

__version__ = "0.1.0"

__all__ = [
    "Sector",
    "OperatorFactor",
    "OperatorTerm",
    "OpSum",
    "Basis",
    "SpinHalfBasis",
    "FermionBasis",
    "HubbardBasis",
    "TJBasis",
    "GenericBasis",
    "Site",
    "SpinHalfSite",
    "FermionSite",
    "HubbardSite",
    "TJSite",
    "BosonSite",
    "HardcoreBosonSite",
    "SpinSSite",
    "MatrixFreeHamiltonian",
    "CUDAHamiltonian",
    "LanczosResult",
    "lanczos_ground_state",
    "DavidsonResult",
    "davidson_lowest",
    "DynamicsResult",
    "continued_fraction_coeffs",
    "evaluate_spectral_function",
    "FTLMResult",
    "ftlm",
]
