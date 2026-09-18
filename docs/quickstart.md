# Quickstart: Heisenberg Model in 5 Minutes

Welcome to **qkrylov**! In this guide, we'll walk you through solving the quantum Heisenberg model step-by-step. By the end of this tutorial, you'll know how to define a Hilbert space, build operators, construct a matrix-free Hamiltonian, and find its ground state.

## Installation

Before we begin, ensure you have installed the library for your preferred environment:

=== "🐍 Python (Pip)"

    === "CPU Only"
        ```bash
        pip install qkrylov
        ```
    === "CUDA 12 (Linux)"
        ```bash
        pip install qkrylov[cuda] --extra-index-url https://sjp95.github.io/qkrylov/whl/cu12
        ```

=== "🔴 Julia (Pkg)"

    === "CPU Only"
        ```julia
        using Pkg
        Pkg.add(url="https://github.com/sjp95/qkrylov.git", subdir="bindings/julia")
        ```
    === "CUDA 12 (Linux)"
        ```julia
        using Pkg
        # Julia automatically detects CUDA on Linux and downloads the GPU binary
        Pkg.add(url="https://github.com/sjp95/qkrylov.git", subdir="bindings/julia")
        ```

=== "⚙️ C++ (Source)"

    === "CPU Only"
        ```bash
        mkdir build && cd build
        cmake .. -DKokkos_ENABLE_OPENMP=ON
        make -j4
        ```
    === "CUDA 12 (Linux)"
        ```bash
        mkdir build && cd build
        cmake .. -DKokkos_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=$(which nvcc)
        make -j4
        ```

## Step 1: Define the Hilbert Space (Basis)

The first step in any exact diagonalization calculation is defining the Hilbert space. In qkrylov, this is done using a `Basis` object. For a spin-1/2 system, we use `SpinHalfBasis`.

We can restrict our Hilbert space to a specific quantum number sector. For the Heisenberg model, the total magnetization ($S_z$) is conserved. Restricting the calculation to a specific $S_z$ sector drastically reduces the dimension of the Hilbert space.

=== "Python"
    ```python
    import qkrylov as qk

    # A system of N=4 sites, restricting to the total Sz = 0 sector.
    N = 4
    basis = qk.SpinHalfBasis(N=N, sz=0)
    
    # Without quantum number sectors, it would just be:
    # full_basis = qk.SpinHalfBasis(N=N)
    ```
=== "C++"
    ```cpp
    #include <qkrylov.hpp>

    int N = 4;
    // Create a sector for Sz = 0
    qk::Sector sector;
    sector.use_sz = true;
    sector.sz = 0;

    qk::SpinHalfBasis basis(N, sector);
    ```
=== "Julia"
    ```julia
    using QuantumKrylov
    N = 4
    # Create a sector for Sz = 0
    basis = SpinHalfBasis(N, Sector(0))
    ```


## Step 2: Define the Local Physics (Site)

Next, we define the local physical degrees of freedom. A `Site` object defines what operators are available at each site. For spin-1/2, `SpinHalfSite` provides the basic spin operators: `Sz`, `Sp` ($S^+$), and `Sm` ($S^-$).

=== "Python"
    ```python
    site = qk.SpinHalfSite()
    ```
=== "C++"
    ```cpp
    qk::SpinHalfSite site;
    ```
=== "Julia"
    ```julia
    # Note: Site definition is handled implicitly in the new Julia API.
    # The basis object inherently knows the local degrees of freedom.
    ```


## Step 3: Build the Hamiltonian (OpSum)

We represent our Hamiltonian as a sum of local operator terms using `OpSum`. 
The Heisenberg chain Hamiltonian is:
$H = J \sum_i \left[ S^z_i S^z_{i+1} + \frac{1}{2} \left( S^+_i S^-_{i+1} + S^-_i S^+_{i+1} \right) \right]$

In qkrylov, we build this up by appending tuples to our `OpSum` object. The tuple convention is: `(coefficient, operator_name_1, site_index_1, operator_name_2, site_index_2, ...)`

=== "Python"
    ```python
    ops = qk.OpSum()
    J = 1.0  # Coupling constant

    for i in range(N - 1):
        # Sz Sz term
        ops += (J, 'Sz', i, 'Sz', i+1)
        # 1/2 (Sp Sm + Sm Sp) terms
        ops += (J * 0.5, 'Sp', i, 'Sm', i+1)
        ops += (J * 0.5, 'Sm', i, 'Sp', i+1)
    ```
=== "C++"
    ```cpp
    qk::OpSum ops;
    double J = 1.0;

    for (int i = 0; i < N - 1; ++i) {
        ops.add({J, "Sz", i, "Sz", i+1});
        ops.add({J * 0.5, "Sp", i, "Sm", i+1});
        ops.add({J * 0.5, "Sm", i, "Sp", i+1});
    }
    ```
=== "Julia"
    ```julia
    ops = OpSum()
    J = 1.0

    for i in 0:(N-2)
        add_term!(ops, J, "Sz", i, "Sz", i+1)
        add_term!(ops, J * 0.5, "Sp", i, "Sm", i+1)
        add_term!(ops, J * 0.5, "Sm", i, "Sp", i+1)
    end
    ```


## Step 4: Build the MatrixFreeHamiltonian

With the basis, site, and operators defined, we can construct the `MatrixFreeHamiltonian`.

**The Zero-Copy Philosophy:** qkrylov operates completely matrix-free. The full Hamiltonian matrix is *never* stored in RAM. Instead, action of the Hamiltonian on a vector is computed on-the-fly. This allows you to simulate much larger systems than standard sparse matrix methods. Numpy arrays in Python are passed directly into C++ without any memory copying.

=== "Python"
    ```python
    H = qk.MatrixFreeHamiltonian(basis, site, ops)

    print(f"Hilbert space dimension: {H.dimension}")

    # You can apply the Hamiltonian directly to a numpy array x
    # import numpy as np
    # x = np.random.rand(H.dimension)
    # y = H.apply(x) 

    # Or get its diagonal elements
    # diag = H.diagonal()
    ```
=== "C++"
    ```cpp
    qk::MatrixFreeHamiltonian H(basis, site, ops);

    std::cout << "Hilbert space dimension: " << H.dimension() << std::endl;

    // std::vector<double> x(H.dimension(), 1.0);
    // std::vector<double> y = H.apply(x);
    ```
=== "Julia"
    ```julia
    H = MatrixFreeHamiltonian(basis, ops)

    # println("Hilbert space dimension: ", dimension(H))
    ```


## Step 5: Solve for Ground State Energy (Lanczos)

To find the ground state, we use the Lanczos algorithm. `lanczos_ground_state` returns the ground state energy and the eigenvector. Let's verify that the ground state energy for a 2-site Heisenberg model is -0.75.

=== "Python"
    ```python
    energy, state = qk.lanczos_ground_state(H)
    print(f"Ground State Energy: {energy}")
    ```
=== "C++"
    ```cpp
    auto [energy, state] = qk::lanczos_ground_state(H);
    std::cout << "Ground State Energy: " << energy << std::endl;
    ```
=== "Julia"
    ```julia
    res = lanczos_ground_state(H)
    println("Ground State Energy: ", res.energy)
    ```


## Step 6: SciPy Integration (Python Only)

While qkrylov's matrix-free engine provides built-in solvers like Lanczos and Davidson, you can also seamlessly integrate it with the Python ecosystem.

`MatrixFreeHamiltonian` can be cast to a SciPy `LinearOperator`, allowing you to use SciPy's sparse linear algebra tools. If your system is small enough, you can also extract the full sparse matrix.

=== "Python"
    ```python
    import scipy.sparse.linalg as sla

    # Use with SciPy's eigensolver
    lin_op = H.aslinearoperator()
    evals, evecs = sla.eigsh(lin_op, k=1, which='SA')
    print(f"SciPy Ground State Energy: {evals[0]}")

    # For very small systems, you can build the explicit sparse matrix
    sparse_mat = H.to_sparse()
    ```
=== "C++"
    !!! note "Python Only Feature"
        SciPy integration is only available in the Python interface.
=== "Julia"
    !!! tip "SciML Ecosystem"
        Instead of SciPy, Julia integrates natively with the SciML ecosystem. 
        See [SciML Integration](getting_started/sciml_julia.md) for how to use `solve(prob, Lanczos())`.
    

## Next Steps

Now that you have a basic understanding of the qkrylov workflow, check out:

- **Advanced Models:** Learn how to simulate Fermion, Hubbard, and t-J models.
- **Dynamical Properties:** See how to calculate spectral functions and continue fractions.
- **Finite Temperature:** Explore Finite Temperature Lanczos Method (FTLM) for thermodynamics.
