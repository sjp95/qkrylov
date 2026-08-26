import numpy as np
import os
from typing import List, Tuple, Union
from . import _qkrylov_cpp as _cpp
from .hamiltonian import MatrixFreeHamiltonian

class LanczosResult:
    """Result of a Lanczos ground state calculation.
    
    Attributes
    ----------
    energy : float
        The computed ground state energy.
    eigenvector : np.ndarray
        The ground state eigenvector.
    """
    def __init__(self, energy: float, eigenvector: np.ndarray):
        self.energy = float(energy)
        self.eigenvector = np.asarray(eigenvector)
        
    def save(self, filename: Union[str, os.PathLike]) -> None:
        """Save LanczosResult to an HDF5 file."""
        import h5py
        with h5py.File(filename, "w") as f:
            f.create_dataset("energy", data=self.energy)
            f.create_dataset("eigenvector", data=self.eigenvector)

    @classmethod
    def load(cls, filename: Union[str, os.PathLike]) -> "LanczosResult":
        """Load LanczosResult from an HDF5 file."""
        import h5py
        with h5py.File(filename, "r") as f:
            energy = float(f["energy"][()])
            eigenvector = np.array(f["eigenvector"])
        return cls(energy=energy, eigenvector=eigenvector)

    def __repr__(self) -> str:
        return f"LanczosResult(energy={self.energy:.10f})"


def lanczos_ground_state(
    H: MatrixFreeHamiltonian, 
    maxiter: int = 200, 
    tol: float = 1e-12
) -> LanczosResult:
    """Find the ground state of a Hamiltonian using the Lanczos algorithm.
    
    Parameters
    ----------
    H : MatrixFreeHamiltonian
        The matrix-free Hamiltonian.
    maxiter : int, optional
        Maximum number of Lanczos iterations (default 200).
    tol : float, optional
        Convergence tolerance (default 1e-12).
        
    Returns
    -------
    LanczosResult
        The ground state energy and eigenvector.
    """
    energy, eigenvector = _cpp.lanczos_ground_state(H._cpp_obj, maxiter, tol)
    return LanczosResult(
        energy=energy,
        eigenvector=eigenvector  # zero-copy NumPy array backed by C++ memory
    )


class DavidsonResult:
    """Result of a Davidson calculation.
    
    Attributes
    ----------
    eigenvalues : np.ndarray
        The lowest eigenvalues.
    eigenvectors : List[np.ndarray]
        The corresponding eigenvectors.
    """
    def __init__(self, eigenvalues: np.ndarray, eigenvectors: List[np.ndarray]):
        self.eigenvalues = np.asarray(eigenvalues, dtype=float)
        self.eigenvectors = [np.asarray(ev) for ev in eigenvectors]
        
    def save(self, filename: Union[str, os.PathLike]) -> None:
        """Save DavidsonResult to an HDF5 file."""
        import h5py
        with h5py.File(filename, "w") as f:
            f.create_dataset("eigenvalues", data=self.eigenvalues)
            grp = f.create_group("eigenvectors")
            for i, ev in enumerate(self.eigenvectors):
                grp.create_dataset(f"vec_{i}", data=ev)

    @classmethod
    def load(cls, filename: Union[str, os.PathLike]) -> "DavidsonResult":
        """Load DavidsonResult from an HDF5 file."""
        import h5py
        with h5py.File(filename, "r") as f:
            eigenvalues = np.array(f["eigenvalues"])
            grp = f["eigenvectors"]
            eigenvectors = [np.array(grp[f"vec_{i}"]) for i in range(len(grp))]
        return cls(eigenvalues=eigenvalues, eigenvectors=eigenvectors)

    def __repr__(self) -> str:
        return f"DavidsonResult(energies={self.eigenvalues})"


def davidson_lowest(
    H: MatrixFreeHamiltonian,
    n_eig: int = 1,
    max_subspace: int = 20,
    tol: float = 1e-8
) -> DavidsonResult:
    """Find the lowest eigenpairs using the Davidson algorithm.
    
    Parameters
    ----------
    H : MatrixFreeHamiltonian
        The matrix-free Hamiltonian.
    n_eig : int, optional
        Number of eigenpairs to compute (default 1).
    max_subspace : int, optional
        Maximum subspace size before restart (default 20).
    tol : float, optional
        Convergence tolerance (default 1e-8).
        
    Returns
    -------
    DavidsonResult
        The lowest eigenvalues and eigenvectors.
    """
    res = _cpp.davidson_lowest(H._cpp_obj, n_eig, max_subspace, tol)
    
    return DavidsonResult(
        eigenvalues=np.array(res.eigenvalues, dtype=float),  # small, copy is fine
        eigenvectors=[np.asarray(ev) for ev in res.eigenvectors]
    )


class DynamicsResult:
    """Result of continued fraction Lanczos."""
    def __init__(self, alphas, betas, norm_phi0):
        self.alphas = np.asarray(alphas)     # zero-copy view from C++
        self.betas  = np.asarray(betas)      # zero-copy view from C++
        self.norm_phi0 = float(norm_phi0)

    def save(self, filename: Union[str, os.PathLike]) -> None:
        """Save DynamicsResult to an HDF5 file."""
        import h5py
        with h5py.File(filename, "w") as f:
            f.create_dataset("alphas", data=self.alphas)
            f.create_dataset("betas", data=self.betas)
            f.create_dataset("norm_phi0", data=self.norm_phi0)

    @classmethod
    def load(cls, filename: Union[str, os.PathLike]) -> "DynamicsResult":
        """Load DynamicsResult from an HDF5 file."""
        import h5py
        with h5py.File(filename, "r") as f:
            alphas = np.array(f["alphas"])
            betas = np.array(f["betas"])
            norm_phi0 = float(f["norm_phi0"][()])
        return cls(alphas=alphas, betas=betas, norm_phi0=norm_phi0)

def continued_fraction_coeffs(
    H: MatrixFreeHamiltonian,
    phi0: np.ndarray,
    n_iter: int = 100
) -> DynamicsResult:
    phi0 = np.ascontiguousarray(phi0, dtype=np.complex128)
    alphas, betas, norm_phi0 = _cpp.continued_fraction_coeffs(H._cpp_obj, phi0, n_iter)
    return DynamicsResult(alphas, betas, norm_phi0)

def evaluate_spectral_function(
    res: DynamicsResult,
    omega: float,
    E0: float,
    eta: float = 0.1
) -> float:
    alphas = np.ascontiguousarray(res.alphas, dtype=np.float64)
    betas  = np.ascontiguousarray(res.betas,  dtype=np.float64)
    return _cpp.evaluate_spectral_function(alphas, betas, res.norm_phi0, omega, E0, eta)


class CorrectionVectorResult:
    """Result of a correction vector spectral calculation."""
    def __init__(self, correction_vector: np.ndarray, spectral_function: float, iterations: int, converged: bool):
        self.correction_vector = np.asarray(correction_vector)
        self.spectral_function = float(spectral_function)
        self.iterations = int(iterations)
        self.converged = bool(converged)

    def save(self, filename: Union[str, os.PathLike]) -> None:
        """Save CorrectionVectorResult to an HDF5 file."""
        import h5py
        with h5py.File(filename, "w") as f:
            f.create_dataset("correction_vector", data=self.correction_vector)
            f.create_dataset("spectral_function", data=self.spectral_function)
            f.create_dataset("iterations", data=self.iterations)
            f.create_dataset("converged", data=self.converged)

    @classmethod
    def load(cls, filename: Union[str, os.PathLike]) -> "CorrectionVectorResult":
        """Load CorrectionVectorResult from an HDF5 file."""
        import h5py
        with h5py.File(filename, "r") as f:
            correction_vector = np.array(f["correction_vector"])
            spectral_function = float(f["spectral_function"][()])
            iterations = int(f["iterations"][()])
            converged = bool(f["converged"][()])
        return cls(correction_vector=correction_vector, spectral_function=spectral_function,
                   iterations=iterations, converged=converged)

def correction_vector_spectral(
    H: MatrixFreeHamiltonian,
    Op_psi0: np.ndarray,
    E0: float,
    omega: float,
    eta: float = 0.1,
    max_iter: int = 500,
    tol: float = 1e-8
) -> CorrectionVectorResult:
    Op_psi0 = np.ascontiguousarray(Op_psi0, dtype=np.complex128)
    vector, spectral_func, iters, converged = _cpp.correction_vector_spectral(
        H._cpp_obj, Op_psi0, E0, omega, eta, max_iter, tol
    )
    return CorrectionVectorResult(vector, spectral_func, iters, converged)


class FTLMResult:
    def __init__(self, cpp_res):
        self.beta = cpp_res.beta
        self.partition_function = cpp_res.partition_function
        self.internal_energy = cpp_res.internal_energy
        self.specific_heat = cpp_res.specific_heat

def ftlm(
    H: MatrixFreeHamiltonian,
    beta: float,
    n_random: int = 50,
    n_steps: int = 100
) -> FTLMResult:
    res = _cpp.ftlm(H._cpp_obj, beta, n_random, n_steps)
    return FTLMResult(res)
