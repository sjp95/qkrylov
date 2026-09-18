# SciML Integration in Julia

Julia bindings for `qkrylov` integrate well with SciML ecosystem. Formulate eigenproblem as standard SciML problem.

## Example: 1D Heisenberg Model

```julia
using qkrylov
using SciMLBase

# Create 1D Heisenberg Hamiltonian (L=10)
basis = SpinBasis(10, 1//2, conserve_Sz=true, Sz=0)
H = build_heisenberg_hamiltonian(basis, J=1.0)

# Define EigenProblem with SciML interface
prob = EigenProblem(H, n_eigenvalues=1, which=:SR)

# Solve with Lanczos
sol = solve(prob, Lanczos())

println("Ground state energy: ", sol.values[1])
```
