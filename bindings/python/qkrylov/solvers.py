import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any, Union, Sequence
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
    def __init__(self, energy: float, eigenvector: np.ndarray, iterations: int = 0, converged: bool = True):
        self.energy = energy
        self.eigenvector = eigenvector
        self.iterations = iterations
        self.converged = converged

    def __iter__(self):
        return iter((self.energy, self.eigenvector))

    def __getitem__(self, idx):
        return (self.energy, self.eigenvector)[idx]

    def __len__(self):
        return 2
        
    def __repr__(self) -> str:
        return f"LanczosResult(energy={self.energy:.10f})"


class Solver(ABC):
    """Abstract base class for all eigensolvers and dynamical solvers."""

    @abstractmethod
    def solve(self, H: MatrixFreeHamiltonian, *args, **kwargs):
        """Execute the solver algorithm on the Hamiltonian."""
        pass

    def __call__(self, H: MatrixFreeHamiltonian, *args, **kwargs):
        """Allow calling solver instance directly: `solver(H)`."""
        return self.solve(H, *args, **kwargs)


class Lanczos(Solver):
    """Standard Single-Pass Lanczos eigensolver.

    Caches all Krylov basis vectors in memory and performs full reorthogonalization.

    Parameters
    ----------
    maxiter : int, optional
        Maximum number of Lanczos iterations (default 200).
    tol : float, optional
        Convergence tolerance (default 1e-12).
    max_iter : int, optional
        Alias for maxiter.
    """

    def __init__(self, maxiter: int = 200, tol: float = 1e-12, max_iter: Optional[int] = None):
        if max_iter is not None:
            maxiter = max_iter
        self.maxiter = int(maxiter)
        self.tol = float(tol)

    @property
    def max_iter(self) -> int:
        return self.maxiter

    @max_iter.setter
    def max_iter(self, val: int):
        self.maxiter = int(val)

    def solve(self, H: MatrixFreeHamiltonian) -> LanczosResult:
        s_dtype = "_FP64" if getattr(H, "dtype", np.float32) == np.float64 else "_FP32"
        fn = getattr(_cpp, f"lanczos_ground_state_{H._backend_suffix}{s_dtype}")
        energy, eigenvector = fn(H._cpp_obj, self.maxiter, self.tol)
        return LanczosResult(energy=energy, eigenvector=eigenvector)


class LanczosTwoPass(Solver):
    """Memory-efficient Two-Pass Lanczos eigensolver for large Hilbert spaces.

    Pass 1 computes the tridiagonal matrix and Ritz values without storing vectors.
    Pass 2 regenerates the ground state Ritz eigenvector on the fly with O(3D) memory.

    Parameters
    ----------
    maxiter : int, optional
        Maximum number of Lanczos iterations (default 200).
    tol : float, optional
        Convergence tolerance (default 1e-12).
    max_iter : int, optional
        Alias for maxiter.
    """

    def __init__(self, maxiter: int = 200, tol: float = 1e-12, max_iter: Optional[int] = None):
        if max_iter is not None:
            maxiter = max_iter
        self.maxiter = int(maxiter)
        self.tol = float(tol)

    @property
    def max_iter(self) -> int:
        return self.maxiter

    @max_iter.setter
    def max_iter(self, val: int):
        self.maxiter = int(val)

    def solve(self, H: MatrixFreeHamiltonian) -> LanczosResult:
        s_dtype = "_FP64" if getattr(H, "dtype", np.float32) == np.float64 else "_FP32"
        fn = getattr(_cpp, f"lanczos_two_pass_{H._backend_suffix}{s_dtype}")
        energy, eigenvector = fn(H._cpp_obj, self.maxiter, self.tol)
        return LanczosResult(energy=energy, eigenvector=eigenvector)


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
    return Lanczos(maxiter=maxiter, tol=tol).solve(H)


def lanczos_two_pass(
    H: MatrixFreeHamiltonian, 
    maxiter: int = 200, 
    tol: float = 1e-12
) -> LanczosResult:
    """Find the ground state using memory-efficient Two-Pass Lanczos algorithm.
    
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
    return LanczosTwoPass(maxiter=maxiter, tol=tol).solve(H)


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

    def __iter__(self):
        return iter((self.eigenvalues, self.eigenvectors))

    def __getitem__(self, idx):
        return (self.eigenvalues, self.eigenvectors)[idx]

    def __len__(self):
        return 2
        
    def __repr__(self) -> str:
        return f"DavidsonResult(energies={self.eigenvalues})"


class Davidson(Solver):
    """Davidson subspace eigensolver for lowest eigenvalues and eigenvectors.

    Parameters
    ----------
    n_eig : int, optional
        Number of lowest eigenpairs to compute (default 1).
    max_subspace : int, optional
        Maximum subspace size before restart (default 20).
    tol : float, optional
        Convergence tolerance (default 1e-8).
    """

    def __init__(self, n_eig: int = 1, max_subspace: int = 20, tol: float = 1e-8):
        self.n_eig = int(n_eig)
        self.max_subspace = int(max_subspace)
        self.tol = float(tol)

    def solve(self, H: MatrixFreeHamiltonian) -> DavidsonResult:
        s_dtype = "_FP64" if getattr(H, "dtype", np.float32) == np.float64 else "_FP32"
        fn = getattr(_cpp, f"davidson_lowest_{H._backend_suffix}{s_dtype}")
        res = fn(H._cpp_obj, self.n_eig, self.max_subspace, self.tol)
        return DavidsonResult(
            eigenvalues=np.array(res.eigenvalues, dtype=float),
            eigenvectors=[np.asarray(ev) for ev in res.eigenvectors]
        )


def davidson_lowest(
    H: MatrixFreeHamiltonian,
    n_eig: int = 1,
    max_subspace: int = 20,
    tol: float = 1e-8
) -> DavidsonResult:
    """Find the lowest eigenpairs using the Davidson algorithm."""
    return Davidson(n_eig=n_eig, max_subspace=max_subspace, tol=tol).solve(H)


class DynamicsResult:
    """Result of continued fraction Lanczos."""
    def __init__(self, alphas, betas, norm_phi0):
        self.alphas = np.asarray(alphas)     # zero-copy view from C++
        self.betas  = np.asarray(betas)      # zero-copy view from C++
        self.norm_phi0 = norm_phi0

    def __iter__(self):
        return iter((self.alphas, self.betas, self.norm_phi0))

    def __getitem__(self, idx):
        return (self.alphas, self.betas, self.norm_phi0)[idx]

    def __len__(self):
        return 3

    def __repr__(self) -> str:
        return f"DynamicsResult(n_steps={len(self.alphas)}, norm_phi0={self.norm_phi0:.6f})"


class ContinuedFraction(Solver):
    """Continued Fraction Lanczos solver for dynamical response functions.

    Parameters
    ----------
    n_iter : int, optional
        Number of Lanczos iterations / continued fraction coefficients (default 100).
    phi0 : np.ndarray, optional
        Initial state vector |phi0> = Op |psi0>. Can also be supplied to solve().
    """

    def __init__(self, n_iter: int = 100, phi0: Optional[np.ndarray] = None):
        self.n_iter = int(n_iter)
        self.phi0 = phi0

    def solve(self, H: MatrixFreeHamiltonian, phi0: Optional[np.ndarray] = None) -> DynamicsResult:
        vec = phi0 if phi0 is not None else self.phi0
        if vec is None:
            raise ValueError("phi0 vector must be provided to ContinuedFraction either at initialization or in solve()")
        vec = np.ascontiguousarray(
            vec,
            dtype=np.complex128 if getattr(H, "dtype", np.float32) == np.float64 else np.complex64
        )
        s_dtype = "_FP64" if getattr(H, "dtype", np.float32) == np.float64 else "_FP32"
        fn = getattr(_cpp, f"continued_fraction_coeffs_{H._backend_suffix}{s_dtype}")
        alphas, betas, norm_phi0 = fn(H._cpp_obj, vec, self.n_iter)
        return DynamicsResult(alphas, betas, norm_phi0)


def continued_fraction_coeffs(
    H: MatrixFreeHamiltonian,
    phi0: np.ndarray,
    n_iter: int = 100
) -> DynamicsResult:
    """Compute continued fraction coefficients for dynamical spectral function."""
    return ContinuedFraction(n_iter=n_iter).solve(H, phi0)


def evaluate_spectral_function(
    res: DynamicsResult,
    omega: float,
    E0: float,
    eta: float = 0.1
) -> float:
    """Evaluate spectral function A(omega) from continued fraction coefficients."""
    alphas = np.ascontiguousarray(res.alphas, dtype=np.float64)
    betas  = np.ascontiguousarray(res.betas,  dtype=np.float64)
    return _cpp.evaluate_spectral_function(alphas, betas, res.norm_phi0, omega, E0, eta)


class FTLMResult:
    """Result of a Finite-Temperature Lanczos Method calculation."""
    def __init__(self, cpp_res, H: MatrixFreeHamiltonian):
        self._cpp_res = cpp_res
        self._H = H
        self.beta = float(cpp_res.beta)
        self.partition_function = float(cpp_res.partition_function)
        self.internal_energy = float(cpp_res.internal_energy)
        self.specific_heat = float(cpp_res.specific_heat)

    def partition_function_at(self, beta: float) -> float:
        """Compute partition function Z(beta) at a given beta without re-running Lanczos."""
        return float(self._cpp_res.partition_function_at(float(beta)))

    def partition_functions(self, betas: Sequence[float]) -> np.ndarray:
        """Compute partition functions Z(beta) across multiple betas."""
        b_vec = [float(b) for b in betas]
        return np.array(self._cpp_res.partition_functions(b_vec), dtype=np.float64)

    def internal_energy_at(self, beta: float) -> float:
        """Compute internal energy <H>_beta at a given beta without re-running Lanczos."""
        return float(self._cpp_res.internal_energy_at(float(beta)))

    def internal_energies(self, betas: Sequence[float]) -> np.ndarray:
        """Compute internal energy <H>_beta across multiple betas."""
        b_vec = [float(b) for b in betas]
        return np.array(self._cpp_res.internal_energies(b_vec), dtype=np.float64)

    def expectation_value(self, A: MatrixFreeHamiltonian, beta: Union[float, Sequence[float]] = None) -> Union[float, np.ndarray]:
        """Compute thermal expectation value <A>_beta for an observable A.

        Parameters
        ----------
        A : MatrixFreeHamiltonian
            The observable operator Hamiltonian.
        beta : float or sequence of floats, optional
            The inverse temperature(s) beta. If None, uses the beta supplied at FTLM initialization.

        Returns
        -------
        float or np.ndarray
            The expectation value <A>_beta or array of expectation values across betas.
        """
        if beta is None:
            beta = self.beta

        if isinstance(beta, (list, tuple, np.ndarray)):
            b_vec = [float(b) for b in beta]
            res = self._cpp_res.expectation_values(A._cpp_obj, b_vec)
            return np.array(res, dtype=np.float64)
        else:
            return float(self._cpp_res.expectation_value(A._cpp_obj, float(beta)))

    def __iter__(self):
        return iter((self.beta, self.partition_function, self.internal_energy, self.specific_heat))

    def __getitem__(self, idx):
        return (self.beta, self.partition_function, self.internal_energy, self.specific_heat)[idx]

    def __len__(self):
        return 4

    def __repr__(self) -> str:
        return (
            f"FTLMResult(beta={self.beta:.4f}, Z={self.partition_function:.6e}, "
            f"E={self.internal_energy:.6f}, C={self.specific_heat:.6f})"
        )


class FTLM(Solver):
    """Finite-Temperature Lanczos Method (FTLM) solver.

    Parameters
    ----------
    beta : float, optional
        Inverse temperature beta = 1 / (k_B * T) (default 1.0).
    n_random : int, optional
        Number of random starting vectors for trace averaging (default 50).
    n_steps : int, optional
        Number of Lanczos expansion steps per sample (default 100).
    """

    def __init__(self, beta: float = 1.0, n_random: int = 50, n_steps: int = 100):
        self.beta = float(beta)
        self.n_random = int(n_random)
        self.n_steps = int(n_steps)

    def solve(self, H: MatrixFreeHamiltonian) -> FTLMResult:
        s_dtype = "_FP64" if getattr(H, "dtype", np.float32) == np.float64 else "_FP32"
        fn = getattr(_cpp, f"ftlm_{H._backend_suffix}{s_dtype}")
        res = fn(H._cpp_obj, self.beta, self.n_random, self.n_steps)
        return FTLMResult(res, H)


def ftlm(
    H: MatrixFreeHamiltonian,
    beta: float = 1.0,
    n_random: int = 50,
    n_steps: int = 100
) -> FTLMResult:
    """Compute finite-temperature thermodynamic observables using FTLM."""
    return FTLM(beta=beta, n_random=n_random, n_steps=n_steps).solve(H)


def ftlm_dynamical_correlation(
    H: MatrixFreeHamiltonian,
    A: MatrixFreeHamiltonian,
    B: MatrixFreeHamiltonian,
    beta: float,
    n_random: int = 50,
    n_steps_thermal: int = 100,
    n_steps_dyn: int = 100,
    omegas: Union[Sequence[float], np.ndarray] = None,
    eta: float = 0.1
) -> np.ndarray:
    """Compute finite-temperature dynamical correlation function S_{AB}(omega, beta).

    Parameters
    ----------
    H : MatrixFreeHamiltonian
        System Hamiltonian.
    A : MatrixFreeHamiltonian
        First observable operator A.
    B : MatrixFreeHamiltonian
        Second observable operator B.
    beta : float
        Inverse temperature.
    n_random : int, optional
        Number of random starting vectors (default 50).
    n_steps_thermal : int, optional
        Lanczos steps for thermalization (default 100).
    n_steps_dyn : int, optional
        Lanczos steps for dynamics (default 100).
    omegas : sequence of floats or np.ndarray
        Frequency grid omegas.
    eta : float, optional
        Broadening parameter (default 0.1).

    Returns
    -------
    np.ndarray
        Dynamical correlation function values across frequency grid omegas.
    """
    if omegas is None:
        raise ValueError("omegas frequency grid must be provided to ftlm_dynamical_correlation")

    omegas_arr = np.ascontiguousarray(omegas, dtype=np.float64)
    s_dtype = "_FP64" if getattr(H, "dtype", np.float32) == np.float64 else "_FP32"
    fn = getattr(_cpp, f"ftlm_dynamical_correlation_{H._backend_suffix}{s_dtype}")
    res = fn(
        H._cpp_obj, A._cpp_obj, B._cpp_obj,
        float(beta), int(n_random), int(n_steps_thermal), int(n_steps_dyn),
        omegas_arr, float(eta)
    )
    return np.asarray(res, dtype=np.float64)


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

    def __getitem__(self, idx):
        return (self.correction_vector, self.spectral_function, self.iterations, self.converged)[idx]

    def __len__(self):
        return 4

    def __repr__(self) -> str:
        return (
            f"CorrectionVectorResult(spectral_function={self.spectral_function:.10e}, "
            f"iterations={self.iterations}, converged={self.converged})"
        )


class CorrectionVector(Solver):
    """Correction Vector dynamical response solver using Conjugate Gradient.

    Solves ((H - E0 - omega)^2 + eta^2) |Y> = eta * Op |psi0>
    and calculates S(omega) = (1/pi) * Re<Op_psi0 | Y>.

    Parameters
    ----------
    e0 : float
        Ground state energy.
    omega : float
        Frequency / energy transfer.
    eta : float, optional
        Broadening factor (default 0.1).
    max_iter : int, optional
        Maximum CG iterations (default 500).
    tol : float, optional
        CG convergence tolerance (default 1e-8).
    op_psi0 : np.ndarray, optional
        State vector Op |psi0>. Can also be supplied to solve().
    """

    def __init__(
        self,
        e0: float,
        omega: float,
        eta: float = 0.1,
        max_iter: int = 500,
        tol: float = 1e-8,
        op_psi0: Optional[np.ndarray] = None,
        maxiter: Optional[int] = None,
    ):
        if maxiter is not None:
            max_iter = maxiter
        self.e0 = float(e0)
        self.omega = float(omega)
        self.eta = float(eta)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.op_psi0 = op_psi0

    def solve(self, H: MatrixFreeHamiltonian, op_psi0: Optional[np.ndarray] = None) -> CorrectionVectorResult:
        vec = op_psi0 if op_psi0 is not None else self.op_psi0
        if vec is None:
            raise ValueError("op_psi0 must be provided to CorrectionVector either at initialization or in solve()")
        vec = np.ascontiguousarray(
            vec,
            dtype=np.complex128 if getattr(H, "dtype", np.float32) == np.float64 else np.complex64
        )
        s_dtype = "_FP64" if getattr(H, "dtype", np.float32) == np.float64 else "_FP32"
        fn = getattr(_cpp, f"correction_vector_spectral_{H._backend_suffix}{s_dtype}")
        corr_vec, spec_fn, iters, conv = fn(
            H._cpp_obj, vec, float(self.e0), float(self.omega), float(self.eta), int(self.max_iter), float(self.tol)
        )
        return CorrectionVectorResult(
            correction_vector=corr_vec,
            spectral_function=float(spec_fn),
            iterations=int(iters),
            converged=bool(conv)
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
    """Compute correction vector and spectral function using conjugate gradient."""
    return CorrectionVector(e0=E0, omega=omega, eta=eta, max_iter=max_iter, tol=tol).solve(H, op_psi0)


correction_vector_spectral = correction_vector
