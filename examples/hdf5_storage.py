import qkrylov

def main():
    N = 4
    basis = qkrylov.SpinHalfBasis(N)
    site = qkrylov.SpinHalfSite()
    os = qkrylov.OpSum()
    for i in range(N - 1):
        os += 1.0, "Sz", i, "Sz", i+1

    H = qkrylov.MatrixFreeHamiltonian(basis, site, os)
    res = qkrylov.lanczos_ground_state(H)

    # Save Lanczos result to HDF5
    res.save("result.h5")
    print("Result saved to result.h5")

    # Load Lanczos result from HDF5
    loaded_res = qkrylov.LanczosResult.load("result.h5")
    print("Loaded energy from HDF5:", loaded_res.energy)

if __name__ == "__main__":
    main()
