#=
 = QKRYLOV FULLY ANNOTATED TUTORIAL
 =#

#=
# Building the Hamiltonian

Welcome to the quickstart. In this tutorial, we will construct a 1D Heisenberg Spin-1/2 chain. 


## Step 1: The Hilbert Space

First, we define our Hilbert space. We use `SpinHalfBasis`.
=#

using QuantumKrylov

b = SpinHalfBasis(10, Sector(0))
println("Basis dimension: ", dimension(b))

#=
## Step 2: Operator Sum

Next, we use `OpSum` to symbolically build our interactions.
=#

ops = OpSum()
J = 1.0
for i in 0:8
    add_term!(ops, J, "Sz", i, "Sz", i + 1)
    add_term!(ops, J*0.5, "Sp", i, "Sm", i + 1)
    add_term!(ops, J*0.5, "Sm", i, "Sp", i + 1)
end

#=
## Step 3: Solve

Finally, we construct the matrix-free Hamiltonian and solve it.
=#

H = MatrixFreeHamiltonian(b, ops)
res = lanczos_ground_state(H)
println("Ground State Energy: ", res.energy)
