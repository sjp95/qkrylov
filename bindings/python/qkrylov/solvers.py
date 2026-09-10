import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
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
        self.energy = energy
        self.eigenvector = eigenvector
        
    def __repr__(self) -> str:
        return f"LanczosResult(energy={self.energy:.10f})"


def lanczos_ground_state(
    H: MatrixFreeHamiltonian, 
    maxiter: int = 200, 
    tol: float = 1e-12,
    two_pass: bool = True
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
    two_pass : bool, optional
        Whether to use two-pass Lanczos (default True).
        
    Returns
    -------
    LanczosResult
        The ground state energy and eigenvector.
    """
    s_dtype = "_FP64" if H.dtype == np.float64 else "_FP32"
    energy, eigenvector = getattr(_cpp, f"lanczos_ground_state_{H._backend_suffix}{s_dtype}")(H._cpp_obj, maxiter, tol, two_pass)
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
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        
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
    s_dtype = "_FP64" if H.dtype == np.float64 else "_FP32"
    res = getattr(_cpp, f"davidson_lowest_{H._backend_suffix}{s_dtype}")(H._cpp_obj, n_eig, max_subspace, tol)
    
    return DavidsonResult(
        eigenvalues=np.array(res.eigenvalues, dtype=float),  # small, copy is fine
        eigenvectors=[np.asarray(ev) for ev in res.eigenvectors]
    )

# For Dynamics and FTLM, we wrap them directly as well.

class DynamicsResult:
    """Result of continued fraction Lanczos."""
    def __init__(self, alphas, betas, norm_phi0):
        self.alphas = np.asarray(alphas)     # zero-copy view from C++
        self.betas  = np.asarray(betas)      # zero-copy view from C++
        self.norm_phi0 = norm_phi0

def continued_fraction_coeffs(
    H: MatrixFreeHamiltonian,
    phi0: np.ndarray,
    n_iter: int = 100
) -> DynamicsResult:
    phi0 = np.ascontiguousarray(phi0, dtype=np.complex128 if getattr(H, "dtype", np.float32) == np.float64 else np.complex64)
    alphas, betas, norm_phi0 = getattr(_cpp, f"continued_fraction_coeffs_{H._backend_suffix}")(H._cpp_obj, phi0, n_iter)
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
    s_dtype = "_FP64" if H.dtype == np.float64 else "_FP32"
    res = getattr(_cpp, f"ftlm_{H._backend_suffix}{s_dtype}")(H._cpp_obj, beta, n_random, n_steps)
    return FTLMResult(res)


@dataclass
class CorrectionVectorResult:
    """Result of a correction vector calculation.

    Attributes
    ----------
    correction_vector : np.ndarray
        The computed correction vector.
    spectral_function : float
        The computed spectral function value S(omega).
    iterations : int
        Number of conjugate gradient iterations.
    converged : bool
        Whether the solver converged within tolerance.
    """
    correction_vector: np.ndarray
    spectral_function: float
    iterations: int
    converged: bool

    def __iter__(self):
        return iter((self.correction_vector, self.spectral_function, self.iterations, self.converged))

    def __repr__(self) -> str:
        return (
            f"CorrectionVectorResult(spectral_function={self.spectral_function:.10e}, "
            f"iterations={self.iterations}, converged={self.converged})"
        )


def correction_vector(
    H: MatrixFreeHamiltonian,
    op_psi0: np.ndarray,
    E0: float,
    omega: float,
    eta: float = 0.1,
    max_iter: int = 500,
    tol: float = 1e-8
) -> CorrectionVectorResult:
    """Compute correction vector and spectral function using conjugate gradient.

    Solves ((H - E0 - omega)^2 + eta^2) |Y> = eta * Op_psi0
    and calculates S(omega) = (1/pi) * Re<Op_psi0 | Y>.

    Parameters
    ----------
    H : MatrixFreeHamiltonian
        The matrix-free Hamiltonian.
    op_psi0 : np.ndarray
        State vector after applying the excitation operator to the ground state.
    E0 : float
        Ground state energy.
    omega : float
        Frequency / energy transfer.
    eta : float, optional
        Broadening factor (default 0.1).
    max_iter : int, optional
        Maximum CG iterations (default 500).
    tol : float, optional
        CG convergence tolerance (default 1e-8).

    Returns
    -------
    CorrectionVectorResult
        Result containing correction_vector, spectral_function, iterations, converged.
    """
    s_dtype = "_FP64" if getattr(H, "dtype", np.float32) == np.float64 else "_FP32"
    op_psi0 = np.ascontiguousarray(
        op_psi0,
        dtype=np.complex128 if getattr(H, "dtype", np.float32) == np.float64 else np.complex64
    )
    cv_func = getattr(_cpp, f"correction_vector_spectral_{H._backend_suffix}{s_dtype}")
    corr_vec, spec_fn, iters, conv = cv_func(
        H._cpp_obj, op_psi0, float(E0), float(omega), float(eta), int(max_iter), float(tol)
    )
    return CorrectionVectorResult(
        correction_vector=corr_vec,
        spectral_function=float(spec_fn),
        iterations=int(iters),
        converged=bool(conv)
    )


correction_vector_spectral = correction_vector

