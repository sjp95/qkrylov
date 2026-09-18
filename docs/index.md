# QKrylov: The Polyglot Quantum Engine

**QKrylov** is a high-performance, matrix-free exact diagonalization engine. At its heart beats a massively parallel C++ Kokkos core that runs on both CPUs and GPUs. But the true power of QKrylov is how it interacts with the world.

We do not believe in second-class bindings. QKrylov is built with **three first-class citizens**: Python, Julia, and C++.

* **Python:** Deeply integrated with PyTorch tensors and object-oriented solver classes.
* **Julia:** Native integration with the SciML ecosystem using Multiple Dispatch and `solve(prob, alg)`.
* **C++:** Zero-overhead C++20 template policies and hardware execution tags.

Write your physics in the language you love. Execute at bare-metal CUDA speeds.

## The First-Class Experience

No matter which language you choose, the API feels entirely native.

=== "Python (PyTorch Style)"
    ```python
    import qkrylov as qk

    # Define physics
    basis = qk.basis.SpinHalf(num_sites=10, sz=0)
    H = qk.Hamiltonian(basis, opsum).to("cuda")

    # OOP Solvers
    solver = qk.solvers.LanczosTwoPass(tol=1e-8)
    evals, evecs = solver.solve(H)
    ```

=== "Julia (SciML Style)"
    ```julia
    using QuantumKrylov

    # Define physics
    basis = Basis(SpinHalf(), 10; Sz = 0)
    H = MatrixFreeHamiltonian(basis, H_sum; device=CUDADevice())

    # SciML Multiple Dispatch
    prob = GroundStateProblem(H)
    sol = solve(prob, Lanczos(variation=TwoPass()))
    ```

=== "C++ (Template Policies)"
    ```cpp
    #include <qkrylov/qkrylov.hpp>

    // Strong Types
    auto basis = basis::SpinHalf(10, basis::sector::Sz{0});
    Hamiltonian H(basis, H_ops, device::gpu{}); 

    // Compile-time tag dispatch
    auto [energy, vec] = solvers::lanczos<solvers::policy::TwoPass>(H, config);
    ```

## Why Matrix-Free?

Traditional solvers like `scipy.sparse.linalg` force you to build the Hamiltonian matrix in memory. For quantum spin chains, memory grows exponentially ($2^L$). At $L=20$, standard solvers crash your machine.

QKrylov is **Matrix-Free**. It applies the Hamiltonian operator bitwise directly onto the state vector on the fly. You never run out of RAM, and you harness the extreme memory bandwidth of modern GPUs.
