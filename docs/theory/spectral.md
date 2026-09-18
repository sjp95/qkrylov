# Spectral Functions & Dynamical Response

Dynamical correlation functions provide direct theoretical predictions for experimental spectroscopies, including inelastic neutron scattering (INS), resonant inelastic X-ray scattering (RIXS), and angle-resolved photoemission spectroscopy (ARPES).

---

## 1. Zero-Temperature Green's Functions

Given a Hamiltonian $H$, ground state $|\psi_0\rangle$, and ground-state energy $E_0$, the zero-temperature retarded Green's function for an operator $\hat{O}$ is:

$$G(\omega) = \langle \psi_0 | \hat{O}^\dagger \frac{1}{\omega + E_0 - H + i\eta} \hat{O} | \psi_0 \rangle$$

The corresponding **spectral function** (or dynamical structure factor) is given by:

$$A(\omega) = -\frac{1}{\pi} \operatorname{Im} G(\omega) = \sum_{n} |\langle n | \hat{O} | \psi_0 \rangle|^2 \frac{\eta/\pi}{(\omega - (E_n - E_0))^2 + \eta^2}$$

In the limit $\eta \to 0^+$, this reproduces the discrete Lehmann representation $\sum_n |\langle n | \hat{O} | \psi_0 \rangle|^2 \delta(\omega - (E_n - E_0))$. The small positive parameter $\eta > 0$ acts as a Lorentzian broadening factor corresponding to experimental finite resolution or lifetime.

---

## 2. Continued Fraction Expansion

To compute $G(z)$ where $z = \omega + E_0 + i\eta$ without diagonalizing the full spectrum of $H$, we initialize the Lanczos algorithm using the perturbed state:

$$|v_1\rangle = \frac{\hat{O} |\psi_0\rangle}{\|\hat{O} |\psi_0\rangle\|}$$

Projecting $z - H$ onto the resulting Krylov subspace yields the tridiagonal matrix $z I - T_m$. Using the recursive Cramer's rule for tridiagonal matrices, the resolvent matrix element $G(z) = \langle \phi_0 | (z - H)^{-1} | \phi_0 \rangle$ expands into a **continued fraction**:

$$G(z) = \frac{\langle \phi_0 | \phi_0 \rangle}{z - \alpha_1 - \cfrac{\beta_1^2}{z - \alpha_2 - \cfrac{\beta_2^2}{z - \alpha_3 - \dots}}}$$

where $\alpha_n$ and $\beta_n$ are the standard Lanczos recurrence coefficients.

### Key Advantages
- **Single Krylov Run**: Once the coefficients $\{\alpha_n, \beta_n\}_{n=1}^M$ are computed, $G(z)$ and $A(\omega)$ can be evaluated across thousands of frequency points $\omega$ in milliseconds.
- **Adjustable Broadening**: The broadening parameter $\eta$ can be interactively varied during post-processing without re-running matrix operations.

---

## 3. The Correction Vector Alternative

While continued fractions are ideal for broad frequency sweeps, high numbers of Lanczos steps ($M > 100$) can develop ghost peaks due to loss of orthogonality in the Krylov basis.

For ultra-high resolution at specific target frequencies $\omega$, `qkrylov` also implements the **correction vector method**:

$$(H - E_0 - \omega - i\eta) |x(\omega, \eta)\rangle = \hat{O} |\psi_0\rangle$$

The spectral function is recovered directly via:

$$A(\omega) = -\frac{1}{\pi} \operatorname{Im} \langle \hat{O} \psi_0 | x(\omega, \eta) \rangle = \frac{\eta}{\pi} \|x(\omega, \eta)\|^2$$

This shifts the problem to an iterative linear system solve, providing certified accuracy within the residual tolerance `tol`.

---

## See Also
- [Dynamics Solver API & Code Examples](../solvers/dynamics.md)
- [Lanczos Algorithm Documentation](../solvers/lanczos.md)
