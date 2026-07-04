import qkrylov
import numpy as np
# import matplotlib.pyplot as plt

def main():
    N = 4
    basis = qkrylov.SpinHalfBasis(N)
    site = qkrylov.SpinHalfSite()

    # Heisenberg model
    os = qkrylov.OpSum()
    for i in range(N - 1):
        # Sz_i Sz_{i+1}
        t = qkrylov.OperatorTerm()
        t.coeff = 1.0
        t.factors = [qkrylov.OperatorFactor("Sz", i), qkrylov.OperatorFactor("Sz", i+1)]
        os.add_term(t)
        # 0.5(Sp_i Sm_{j} + Sm_i Sp_j)
        t = qkrylov.OperatorTerm()
        t.coeff = 0.5
        t.factors = [qkrylov.OperatorFactor("Sp", i), qkrylov.OperatorFactor("Sm", i+1)]
        os.add_term(t)
        t = qkrylov.OperatorTerm()
        t.coeff = 0.5
        t.factors = [qkrylov.OperatorFactor("Sm", i), qkrylov.OperatorFactor("Sp", i+1)]
        os.add_term(t)

    H = qkrylov.MatrixFreeHamiltonian(basis, site, os)
    gs = qkrylov.lanczos_ground_state(H)

    print(f"Ground state energy: {gs.energy}")

    # Initial vector phi0 = A |gs> where A = Sz_0
    phi0 = list(gs.eigenvector)

    res = qkrylov.continued_fraction_coeffs(H, phi0, n_iter=20)

    omegas = np.linspace(0, 5, 100)
    S = [qkrylov.evaluate_spectral_function(res, w, gs.energy, eta=0.1) for w in omegas]

    print(f"Spectral function calculated. First few values: {S[:5]}")

if __name__ == "__main__":
    main()
