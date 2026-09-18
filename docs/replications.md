# Replicated Papers

QKrylov is designed for extreme accuracy and precision in quantum many-body physics. To verify the integrity of the engine, we replicate classic analytical results and benchmark them against the exact diagonalization output.

## 1. 1D Heisenberg Model (Bethe Ansatz)

The 1D spin-1/2 Heisenberg model with periodic boundary conditions is exactly solvable in the thermodynamic limit ($L \to \infty$) using the **Bethe Ansatz**. The ground state energy per site is known to be exactly:

$$ \frac{E_0}{L} = \frac{1}{4} - \ln(2) \approx -0.44314718 $$

Below, we use QKrylov to diagonalize finite chains and observe the finite-size scaling towards the exact analytical limit.

=== "Python"
    ```python
    import numpy as np
    from qkrylov import SpinHalfBasis, SpinHalfSite, OpSum, MatrixFreeHamiltonian, lanczos_ground_state
    from qkrylov.operators import Sz, Sp, Sm

    def compute_energy(L):
        # Build a 1D Heisenberg model OpSum
        ops = OpSum(dtype=np.float64)
        for i in range(L - 1):
            ops += 1.0 * Sz(i) * Sz(i + 1) + 0.5 * Sp(i) * Sm(i + 1) + 0.5 * Sm(i) * Sp(i + 1)
        ops += 1.0 * Sz(L - 1) * Sz(0) + 0.5 * Sp(L - 1) * Sm(0) + 0.5 * Sm(L - 1) * Sp(0)
        
        basis = SpinHalfBasis(L, dtype=np.float64)
        site = SpinHalfSite(dtype=np.float64)
        H = MatrixFreeHamiltonian(basis, site, ops, device="cpu", dtype=np.float64)
        result = lanczos_ground_state(H, maxiter=200, tol=1e-10)
        return result.energy / L

    bethe_exact = 0.25 - np.log(2)
    print(f"Exact Limit: {bethe_exact:.6f}")

    for L in [10, 16, 20]:
        e_finite = compute_energy(L)
        print(f"L={L:<2} | Energy/site: {e_finite:.6f} | Finite-Size Error: {abs(e_finite - bethe_exact):.6f}")
    ```

=== "Julia"
    ```julia
    using QKrylov
    using SciMLBase

    function compute_energy(L)
        ops = OpSum(Float64)
        for i in 0:L-2
            ops += 1.0 * Sz(i) * Sz(i + 1) + 0.5 * Sp(i) * Sm(i + 1) + 0.5 * Sm(i) * Sp(i + 1)
        end
        ops += 1.0 * Sz(L - 1) * Sz(0) + 0.5 * Sp(L - 1) * Sm(0) + 0.5 * Sm(L - 1) * Sp(0)

        basis = SpinHalfBasis(L, Float64)
        site = SpinHalfSite(Float64)
        H = MatrixFreeHamiltonian(basis, site, ops, "cpu", Float64)
        
        prob = EigenProblem(H)
        result = solve(prob, Lanczos(maxiter=200, tol=1e-10))
        return result.energy / L
    end

    bethe_exact = 0.25 - log(2)
    println("Exact Limit: ", round(bethe_exact, digits=6))

    for L in [10, 16, 20]
        e_finite = compute_energy(L)
        err = abs(e_finite - bethe_exact)
        println("L=$L | Energy/site: ", round(e_finite, digits=6), " | Finite-Size Error: ", round(err, digits=6))
    end
    ```
