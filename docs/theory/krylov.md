# Krylov Subspace Methods

Krylov subspace methods are the gold standard for extracting extremal eigenvalues, dynamical response, and thermodynamic properties of large-scale sparse and matrix-free quantum systems.

---

## 1. The Krylov Subspace

Given a Hamiltonian $H$ of dimension $D$ and an initial normalized trial vector $v_1$, the $m$-dimensional **Krylov subspace** is defined as:

$$\mathcal{K}_m(H, v_1) = \operatorname{span} \{ v_1, H v_1, H^2 v_1, \dots, H^{m-1} v_1 \}$$

Repeated multiplication by $H$ exponentially amplifies the components of $v_1$ corresponding to extremal eigenvalues (the ground state $E_0$ and highest excited states $E_{\max}$).

---

## 2. The Lanczos Algorithm & Tridiagonalization

Because $H$ is Hermitian ($H = H^\dagger$), the projection of $H$ onto an orthonormal basis $V_m = [v_1, v_2, \dots, v_m]$ of $\mathcal{K}_m(H, v_1)$ reduces to a symmetric **tridiagonal matrix**:

$$T_m = V_m^\dagger H V_m = \begin{pmatrix}
\alpha_1 & \beta_1 & & & 0 \\
\beta_1 & \alpha_2 & \beta_2 & & \\
& \beta_2 & \alpha_3 & \ddots & \\
& & \ddots & \ddots & \beta_{m-1} \\
0 & & & \beta_{m-1} & \alpha_m
\end{pmatrix}$$

The basis vectors $v_j$ satisfy the famous **three-term recurrence**:

$$H v_j = \beta_{j-1} v_{j-1} + \alpha_j v_j + \beta_j v_{j+1}$$

where:
$$\alpha_j = \langle v_j | H | v_j \rangle, \quad w_j = H v_j - \alpha_j v_j - \beta_{j-1} v_{j-1}, \quad \beta_j = \|w_j\|_2, \quad v_{j+1} = \frac{w_j}{\beta_j}$$

Because each step only references $v_{j-1}$, $v_j$, and $v_{j+1}$, the ground-state eigenvalue can be computed with only **3 vectors in memory**.

---

## 3. Rayleigh-Ritz Approximation & Convergence

Diagonalizing the small $m \times m$ matrix $T_m$:

$$T_m y_k = \theta_k y_k \quad (k = 1, \dots, m)$$

yields the **Ritz values** $\theta_k \approx E_k$ and **Ritz vectors**:

$$|u_k\rangle = V_m y_k = \sum_{j=1}^m y_{k, j} |v_j\rangle$$

By the **Kaniel-Paige-Saad theorem**, the error in the ground-state eigenvalue converges exponentially:

$$\theta_1 - E_0 \le (E_{\max} - E_0) \left[ \frac{\tan \angle(v_1, \psi_0)}{T_{m-1}(1 + 2\gamma)} \right]^2 \approx \mathcal{O}\left( e^{-4 m \sqrt{\gamma}} \right)$$

where $\gamma = \frac{E_1 - E_0}{E_{\max} - E_1}$ is the relative spectral gap and $T_k$ is the Chebyshev polynomial of degree $k$.

---

## 4. Finite Precision & Orthogonality

In exact arithmetic, the Lanczos basis vectors $v_j$ are mutually orthogonal ($\langle v_i | v_j \rangle = \delta_{ij}$). In finite-precision floating-point arithmetic (IEEE 754), round-off errors accumulate. As soon as a Ritz value converges, orthogonality rapidly degrades, causing duplicate "ghost" eigenvalues to appear in the spectrum.

### Solutions Implemented in `qkrylov`

1. **Reorthogonalization Policies**:
   - `OnePass_DKGS`: Dynamic Kahan-Gram-Schmidt reorthogonalization prevents loss of orthogonality when computing low-lying excited states.
   - `OnePass_full`: Full reorthogonalization against all vectors $V_m$.
2. **Two-Pass Lanczos (`TwoPass`)**:
   - Computes eigenvalues in pass 1, then regenerates the Ritz vector in pass 2 using only 3 working vectors. Ideal for large ground-state calculations where storing $V_m$ exceeds available RAM.
3. **Davidson Algorithm**:
   - Instead of building a Krylov polynomial subspace, Davidson expands the search subspace with preconditioned residual directions $q_i = (D - \theta_i I)^{-1} r_i$, making it the algorithm of choice for multiple excited states.
