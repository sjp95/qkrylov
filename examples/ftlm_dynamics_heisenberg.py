"""Example demonstrating Finite-Temperature Lanczos Method (FTLM) observables
and finite-temperature dynamical correlation functions S_AB(omega, beta)
for a 1D Heisenberg spin chain using qkrylov.
"""

import numpy as np
import qkrylov as qk

def main():
    print("=== qkrylov FTLM & Dynamical Correlations Example ===")

    N = 4
    basis = qk.basis.SpinHalf(N=N)
    site = qk.site.SpinHalf()

    # 1. Build Hamiltonian H = sum_i (Sz_i Sz_{i+1} + 0.5 (Sp_i Sm_{i+1} + Sm_i Sp_{i+1}))
    os_H = qk.OpSum()
    for i in range(N):
        j = (i + 1) % N
        os_H += 1.0 * qk.Sz(i) * qk.Sz(j) + 0.5 * (qk.Sp(i) * qk.Sm(j) + qk.Sm(i) * qk.Sp(j))

    H = qk.MatrixFreeHamiltonian(basis, site, os_H)

    # 2. Build Observable Mz^2 = (sum_i Sz_i)^2
    os_Mz2 = qk.OpSum()
    for i in range(N):
        for j in range(N):
            os_Mz2 += 1.0 * qk.Sz(i) * qk.Sz(j)

    Mz2 = qk.MatrixFreeHamiltonian(basis, site, os_Mz2)

    # 3. Perform FTLM sampling once
    n_random = 200
    n_steps = 16
    print(f"Running FTLM sampling with {n_random} random vectors and {n_steps} Lanczos steps...")
    ftlm_res = qk.ftlm(H, beta=1.0, n_random=n_random, n_steps=n_steps)

    # 4. Evaluate <Mz^2>_beta across multiple temperatures
    betas = [0.1, 0.5, 1.0, 2.0]
    mz2_vals = ftlm_res.expectation_value(Mz2, betas)

    print("\nThermodynamic Expectation Value <Mz^2>_beta:")
    for b, val in zip(betas, mz2_vals):
        print(f"  beta = {b:3.1f}: <Mz^2> = {val:.6f}")

    # 5. Compute Finite-Temperature Dynamical Correlation S_zz(q=pi, omega)
    # S_zz(q, omega) with q = pi (staggered operator A = B = sum_j (-1)^j Sz_j)
    os_Stag = qk.OpSum()
    for j in range(N):
        os_Stag += ((-1.0)**j) * qk.Sz(j)

    A = qk.MatrixFreeHamiltonian(basis, site, os_Stag)

    omegas = np.linspace(0.0, 4.0, 50)
    beta = 1.0
    print(f"\nComputing S_zz(q=pi, omega) at beta = {beta}...")
    spec = qk.ftlm_dynamical_correlation(
        H, A, A, beta=beta, n_random=100, n_steps_thermal=14, n_steps_dyn=14, omegas=omegas, eta=0.1
    )

    print("\nSpectral Density S(omega) sample values:")
    for w, s in zip(omegas[::10], spec[::10]):
        print(f"  omega = {w:4.2f}: S = {s:.6f}")

    print("\nFTLM example completed successfully!")

if __name__ == "__main__":
    main()
