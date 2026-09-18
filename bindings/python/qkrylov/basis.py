import numpy as np
from typing import Optional
from . import _qkrylov_cpp as _cpp

class Basis:
    """Base class for all Hilbert space bases."""
    
    def __init__(self):
        self._cpp_obj = None

    @property
    def size(self) -> int:
        """The total dimension of this basis."""
        return self._cpp_obj.size()

    @property
    def nsites(self) -> int:
        """The number of sites in this basis."""
        return self._cpp_obj.nsites()

def _build_sector(dtype,

    conserve_sz: bool = False, sz: int = 0,
    conserve_nup: bool = False, nup: int = 0,
    conserve_ndn: bool = False, ndn: int = 0,
    conserve_n: bool = False, n: int = 0,
    conserve_nb: bool = False, nb: int = 0
) -> "_cpp.Sector":
    """Helper to build a C++ Sector object from Python kwargs."""
    suffix = "_FP64" if dtype == np.float64 else "_FP32"
    sec = getattr(_cpp, f"Sector{suffix}")()
    if conserve_sz or sz != 0:
        sec.use_sz = True
        sec.sz2 = sz * 2  # C++ uses 2*Sz
    if conserve_nup or nup != 0:
        sec.use_nup = True
        sec.nup = nup
    if conserve_ndn or ndn != 0:
        sec.use_ndn = True
        sec.ndn = ndn
    if conserve_n or n != 0:
        sec.use_n = True
        sec.n = n
    if conserve_nb or nb != 0:
        sec.use_nb = True
        sec.nb = nb
    return sec


class SpinHalfBasis(Basis):
    """Basis for Spin-1/2 systems.
    
    Parameters
    ----------
    N : int
        Number of spin sites.
    conserve_sz : bool, optional
        Whether to conserve total Sz. If True, only states with the specified `sz` are kept.
    sz : float or int, optional
        The target total Sz sector (default 0). Note: The underlying C++ code uses 2*Sz, 
        so integer or half-integer values are allowed.
    """
    
    def __init__(self, N: int, conserve_sz: bool = False, sz: Optional[float] = None, dtype=np.float32):
        # If sz is explicitly provided, it implies conservation
        if sz is not None:
            conserve_sz = True
        elif conserve_sz and sz is None:
            sz = 0  # default sector when conserve_sz=True but no sz given
        else:
            sz = 0  # no conservation, sz value doesn't matter

        # C++ Sector takes sz2 (which is 2 * sz)
        suffix = "_FP64" if dtype == np.float64 else "_FP32"
        sec = getattr(_cpp, f"Sector{suffix}")()
        if conserve_sz:
            sec.use_sz = True
            sec.sz2 = int(2 * sz)

        self._cpp_obj = getattr(_cpp, f"SpinHalfBasis{suffix}")(N, sec)
        self._conserve_sz = conserve_sz
        self._sz = sz

    def __repr__(self) -> str:
        sec_str = f", sz={self._sz}" if self._conserve_sz else ""
        return f"SpinHalfBasis(N={self.nsites}, dim={self.size}{sec_str})"


class SpinSBasis(Basis):
    """Basis for arbitrary spin S systems.
    
    Parameters
    ----------
    N : int
        Number of spin sites.
    S : float, optional
        Spin quantum number S (default 0.5).
    conserve_sz : bool, optional
        Whether to conserve total Sz. If True, only states with the specified `sz` are kept.
    sz : float or int, optional
        The target total Sz sector (default 0). Note: The underlying C++ code uses 2*Sz, 
        so integer or half-integer values are allowed.
    dtype : np.dtype, optional
        Precision type (np.float32 or np.float64, default np.float32).
    """
    
    def __init__(
        self,
        N: int,
        S: float = 0.5,
        conserve_sz: bool = False,
        sz: Optional[float] = None,
        dtype=np.float32
    ):
        if sz is not None:
            conserve_sz = True
        elif conserve_sz and sz is None:
            sz = 0
        else:
            sz = 0

        suffix = "_FP64" if dtype == np.float64 else "_FP32"
        sec = getattr(_cpp, f"Sector{suffix}")()
        if conserve_sz:
            sec.use_sz = True
            sec.sz2 = int(round(2 * sz))

        self._cpp_obj = getattr(_cpp, f"SpinSBasis{suffix}")(N, float(S), sec)
        self._S = float(S)
        self._conserve_sz = conserve_sz
        self._sz = sz

    @property
    def spin(self) -> float:
        """The spin quantum number S."""
        return self._cpp_obj.spin

    @property
    def dimension_per_site(self) -> int:
        """The local Hilbert space dimension 2S + 1."""
        return self._cpp_obj.dimension_per_site

    def state(self, i: int) -> int:
        """Get the basis state at index i."""
        return self._cpp_obj.state(i)

    def index(self, s: int) -> int:
        """Find the index of basis state s."""
        return self._cpp_obj.index(s)

    def contains(self, s: int) -> bool:
        """Check if basis state s is in this basis."""
        return self._cpp_obj.contains(s)

    def __repr__(self) -> str:
        sec_str = f", sz={self._sz}" if self._conserve_sz else ""
        return f"SpinSBasis(N={self.nsites}, S={self._S}, dim={self.size}{sec_str})"


class FermionBasis(Basis):
    """Basis for spinless fermions.
    
    Parameters
    ----------
    N : int
        Number of fermion sites.
    conserve_n : bool, optional
        Whether to conserve total particle number.
    n : int, optional
        The target total particle number sector (default 0).
    """
    def __init__(self, N: int, conserve_n: bool = False, n: int = 0, dtype=np.float32):
        if n != 0:
            conserve_n = True
        
        suffix = "_FP64" if dtype == np.float64 else "_FP32"
        sec = getattr(_cpp, f"Sector{suffix}")()
        if conserve_n:
            sec.use_n = True
            sec.n = n

        self._cpp_obj = getattr(_cpp, f"FermionBasis{suffix}")(N, sec)
        self._conserve_n = conserve_n
        self._n = n

    def __repr__(self) -> str:
        sec_str = f", n={self._n}" if self._conserve_n else ""
        return f"FermionBasis(N={self.nsites}, dim={self.size}{sec_str})"


class HubbardBasis(Basis):
    """Basis for interacting electrons (Hubbard model).
    
    Parameters
    ----------
    N : int
        Number of sites.
    conserve_nup : bool, optional
        Whether to conserve number of up-spin electrons.
    nup : int, optional
        Target number of up-spin electrons.
    conserve_ndn : bool, optional
        Whether to conserve number of down-spin electrons.
    ndn : int, optional
        Target number of down-spin electrons.
    """
    def __init__(self, N: int, conserve_nup: bool = False, nup: int = 0, 
                 conserve_ndn: bool = False, ndn: int = 0, dtype=np.float32):
        if nup != 0: conserve_nup = True
        if ndn != 0: conserve_ndn = True

        suffix = "_FP64" if dtype == np.float64 else "_FP32"
        sec = getattr(_cpp, f"Sector{suffix}")()
        if conserve_nup:
            sec.use_nup = True
            sec.nup = nup
        if conserve_ndn:
            sec.use_ndn = True
            sec.ndn = ndn

        self._cpp_obj = getattr(_cpp, f"HubbardBasis{suffix}")(N, sec)
        self._conserve_nup = conserve_nup
        self._conserve_ndn = conserve_ndn
        self._nup = nup
        self._ndn = ndn

    def __repr__(self) -> str:
        sec_strs = []
        if self._conserve_nup: sec_strs.append(f"nup={self._nup}")
        if self._conserve_ndn: sec_strs.append(f"ndn={self._ndn}")
        sec_str = ", " + ", ".join(sec_strs) if sec_strs else ""
        return f"HubbardBasis(N={self.nsites}, dim={self.size}{sec_str})"


class TJBasis(Basis):
    """Basis for t-J model (doped antiferromagnet)."""
    def __init__(self, N: int, conserve_nup: bool = False, nup: int = 0, 
                 conserve_ndn: bool = False, ndn: int = 0, dtype=np.float32):
        if nup != 0: conserve_nup = True
        if ndn != 0: conserve_ndn = True

        suffix = "_FP64" if dtype == np.float64 else "_FP32"
        sec = getattr(_cpp, f"Sector{suffix}")()
        if conserve_nup:
            sec.use_nup = True
            sec.nup = nup
        if conserve_ndn:
            sec.use_ndn = True
            sec.ndn = ndn

        self._cpp_obj = getattr(_cpp, f"TJBasis{suffix}")(N, sec)
        self._conserve_nup = conserve_nup
        self._conserve_ndn = conserve_ndn
        self._nup = nup
        self._ndn = ndn

    def __repr__(self) -> str:
        sec_strs = []
        if self._conserve_nup: sec_strs.append(f"nup={self._nup}")
        if self._conserve_ndn: sec_strs.append(f"ndn={self._ndn}")
        sec_str = ", " + ", ".join(sec_strs) if sec_strs else ""
        return f"TJBasis(N={self.nsites}, dim={self.size}{sec_str})"


# Aliases conforming to API blueprint
SpinHalf = SpinHalfBasis
SpinS = SpinSBasis
Fermion = FermionBasis
Hubbard = HubbardBasis
TJ = TJBasis
