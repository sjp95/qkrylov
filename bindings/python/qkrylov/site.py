from . import _qkrylov_cpp as _cpp

import numpy as np

class Site:
    """Base class for all local site physics."""
    
    def __init__(self, dtype=np.float32):
        suffix = '_FP64' if dtype == np.float64 else '_FP32'
        self._cpp_obj = None

class SpinHalfSite(Site):
    """Local site physics for Spin-1/2 systems.
    
    Supports operators: 'Sz', 'Sp', 'Sm', 'Sx', 'Sy'
    """
    def __init__(self, dtype=np.float32):
        suffix = '_FP64' if dtype == np.float64 else '_FP32'
        self._cpp_obj = getattr(_cpp, f'SpinHalfSite{suffix}')()

    def __repr__(self) -> str:
        return "SpinHalfSite()"

class SpinSSite(Site):
    """Local site physics for arbitrary spin S systems.
    
    Supports operators: 'Sz', 'Sp', 'Sm'
    """
    def __init__(self, S: float = 0.5, dtype=np.float32):
        suffix = '_FP64' if dtype == np.float64 else '_FP32'
        self._cpp_obj = getattr(_cpp, f'SpinSSite{suffix}')(float(S))
        self._S = float(S)
        self._dtype = dtype

    @property
    def spin(self) -> float:
        """The spin quantum number S."""
        return self._cpp_obj.spin

    @property
    def dimension_per_site(self) -> int:
        """The local Hilbert space dimension 2S + 1."""
        return self._cpp_obj.dimension_per_site

    def __repr__(self) -> str:
        return f"SpinSSite(S={self._S})"

class FermionSite(Site):
    """Local site physics for spinless fermions."""
    def __init__(self, dtype=np.float32):
        suffix = '_FP64' if dtype == np.float64 else '_FP32'
        self._cpp_obj = getattr(_cpp, f'FermionSite{suffix}')()
        
    def __repr__(self) -> str:
        return "FermionSite()"

class HubbardSite(Site):
    """Local site physics for interacting electrons (spin-1/2 fermions)."""
    def __init__(self, dtype=np.float32):
        suffix = '_FP64' if dtype == np.float64 else '_FP32'
        self._cpp_obj = getattr(_cpp, f'HubbardSite{suffix}')()

    def __repr__(self) -> str:
        return "HubbardSite()"

class TJSite(Site):
    """Local site physics for t-J model (doped antiferromagnets)."""
    def __init__(self, dtype=np.float32):
        suffix = '_FP64' if dtype == np.float64 else '_FP32'
        self._cpp_obj = getattr(_cpp, f'TJSite{suffix}')()

    def __repr__(self) -> str:
        return "TJSite()"


# Aliases conforming to API blueprint
SpinHalf = SpinHalfSite
SpinS = SpinSSite
Fermion = FermionSite
Hubbard = HubbardSite
TJ = TJSite
