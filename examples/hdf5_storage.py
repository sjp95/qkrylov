import qkrylov
import h5py
import numpy as np

def save_result_h5(filename, result):
    with h5py.File(filename, "w") as f:
        if isinstance(result, qkrylov.LanczosResult):
            f.create_dataset("energy", data=result.energy)
            f.create_dataset("eigenvector", data=np.array(result.eigenvector))
        elif isinstance(result, qkrylov.DavidsonResult):
            f.create_dataset("eigenvalues", data=np.array(result.eigenvalues))
            for i, ev in enumerate(result.eigenvectors):
                f.create_dataset(f"eigenvector_{i}", data=np.array(ev))

def main():
    N = 4
    basis = qkrylov.SpinHalfBasis(N)
    site = qkrylov.SpinHalfSite()
    os = qkrylov.OpSum()
    for i in range(N - 1):
        os += 1.0, "Sz", i, "Sz", i+1

    H = qkrylov.MatrixFreeHamiltonian(basis, site, os)
    res = qkrylov.lanczos_ground_state(H)

    save_result_h5("result.h5", res)
    print("Result saved to result.h5")

if __name__ == "__main__":
    main()
