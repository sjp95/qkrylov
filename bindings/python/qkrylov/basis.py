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

def _build_sector(
    conserve_sz: bool = False, sz: int = 0,
    conserve_nup: bool = False, nup: int = 0,
    conserve_ndn: bool = False, ndn: int = 0,
    conserve_n: bool = False, n: int = 0,
    conserve_nb: bool = False, nb: int = 0
) -> _cpp.Sector:
    """Helper to build a C++ Sector object from Python kwargs."""
    sec = _cpp.Sector()
    if conserve_sz or sz != 0:
        sec.use_sz = True
        sec.sz2 = int(2 * sz)  # C++ uses 2*Sz
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
    """Basis for Spin-1/2 systems."""
    
    def __init__(self, N: int, conserve_sz: bool = False, sz: Optional[float] = None, sector: Optional[_cpp.Sector] = None):
        if sector is not None:
            self._cpp_obj = _cpp.SpinHalfBasis(N, sector)
            self._conserve_sz = sector.use_sz
            self._sz = sector.sz2 / 2.0
            return

        if sz is not None:
            conserve_sz = True
        elif conserve_sz and sz is None:
            sz = 0
        else:
            sz = 0

        sec = _cpp.Sector()
        if conserve_sz:
            sec.use_sz = True
            sec.sz2 = int(2 * sz)

        self._cpp_obj = _cpp.SpinHalfBasis(N, sec)
        self._conserve_sz = conserve_sz
        self._sz = sz

    def __repr__(self) -> str:
        sec_str = f", sz={self._sz}" if self._conserve_sz else ""
        return f"SpinHalfBasis(N={self.nsites}, dim={self.size}{sec_str})"


class SpinSBasis(Basis):
    """Basis for arbitrary Spin-S systems."""
    def __init__(self, N: int, S: float = 0.5, conserve_sz: bool = False, sz: Optional[float] = None, sector: Optional[_cpp.Sector] = None):
        if sector is not None:
            self._cpp_obj = _cpp.SpinSBasis(N, S, sector)
            self._conserve_sz = sector.use_sz
            self._sz = sector.sz2 / 2.0
            return

        if sz is not None:
            conserve_sz = True
        elif conserve_sz and sz is None:
            sz = 0
        else:
            sz = 0

        sec = _cpp.Sector()
        if conserve_sz:
            sec.use_sz = True
            sec.sz2 = int(2 * sz)

        self._cpp_obj = _cpp.SpinSBasis(N, S, sec)
        self._conserve_sz = conserve_sz
        self._sz = sz


class FermionBasis(Basis):
    """Basis for spinless fermions."""
    def __init__(self, N: int, conserve_n: bool = False, n: int = 0, sector: Optional[_cpp.Sector] = None):
        if sector is not None:
            self._cpp_obj = _cpp.FermionBasis(N, sector)
            self._conserve_n = sector.use_n
            self._n = sector.n
            return

        if n != 0:
            conserve_n = True
        
        sec = _cpp.Sector()
        if conserve_n:
            sec.use_n = True
            sec.n = n

        self._cpp_obj = _cpp.FermionBasis(N, sec)
        self._conserve_n = conserve_n
        self._n = n

    def __repr__(self) -> str:
        sec_str = f", n={self._n}" if self._conserve_n else ""
        return f"FermionBasis(N={self.nsites}, dim={self.size}{sec_str})"


class HubbardBasis(Basis):
    """Basis for interacting electrons (Hubbard model)."""
    def __init__(self, N: int, conserve_nup: bool = False, nup: int = 0, 
                 conserve_ndn: bool = False, ndn: int = 0, sector: Optional[_cpp.Sector] = None):
        if sector is not None:
            self._cpp_obj = _cpp.HubbardBasis(N, sector)
            self._conserve_nup = sector.use_nup
            self._conserve_ndn = sector.use_ndn
            self._nup = sector.nup
            self._ndn = sector.ndn
            return

        if nup != 0: conserve_nup = True
        if ndn != 0: conserve_ndn = True

        sec = _cpp.Sector()
        if conserve_nup:
            sec.use_nup = True
            sec.nup = nup
        if conserve_ndn:
            sec.use_ndn = True
            sec.ndn = ndn

        self._cpp_obj = _cpp.HubbardBasis(N, sec)
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
                 conserve_ndn: bool = False, ndn: int = 0, sector: Optional[_cpp.Sector] = None):
        if sector is not None:
            self._cpp_obj = _cpp.TJBasis(N, sector)
            self._conserve_nup = sector.use_nup
            self._conserve_ndn = sector.use_ndn
            self._nup = sector.nup
            self._ndn = sector.ndn
            return

        if nup != 0: conserve_nup = True
        if ndn != 0: conserve_ndn = True

        sec = _cpp.Sector()
        if conserve_nup:
            sec.use_nup = True
            sec.nup = nup
        if conserve_ndn:
            sec.use_ndn = True
            sec.ndn = ndn

        self._cpp_obj = _cpp.TJBasis(N, sec)
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
