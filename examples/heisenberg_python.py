import qkrylov
import numpy as np

def main():
    N = 4
    # Create basis and site
    basis = qkrylov.SpinHalfBasis(N)
    site = qkrylov.SpinHalfSite()

    # Create Heisenberg Hamiltonian
    os = qkrylov.OpSum()
    for i in range(N - 1):
        # Sz_i Sz_{i+1}
        term1 = qkrylov.OperatorTerm()
        term1.coeff = 1.0
        term1.factors = [qkrylov.OperatorFactor("Sz", i), qkrylov.OperatorFactor("Sz", i+1)]
        os.add_term(term1)

        # 0.5 * Sp_i Sm_{i+1}
        term2 = qkrylov.OperatorTerm()
        term2.coeff = 0.5
        term2.factors = [qkrylov.OperatorFactor("Sp", i), qkrylov.OperatorFactor("Sm", i+1)]
        os.add_term(term2)

        # 0.5 * Sm_i Sp_{i+1}
        term3 = qkrylov.OperatorTerm()
        term3.coeff = 0.5
        term3.factors = [qkrylov.OperatorFactor("Sm", i), qkrylov.OperatorFactor("Sp", i+1)]
        os.add_term(term3)

    H = qkrylov.MatrixFreeHamiltonian(basis, site, os)

    # Solve using Lanczos
    result = qkrylov.lanczos_ground_state(H)
    print(f"Lanczos Ground state energy: {result.energy}")

    # Solve using Davidson
    dav_result = qkrylov.davidson_lowest(H, n_eig=1)
    print(f"Davidson Ground state energy: {dav_result.eigenvalues[0]}")

if __name__ == "__main__":
    main()
