# [ID: init_basis]
using QuantumKrylov

b = SpinHalfBasis(10, Sector(0))
println("Basis dimension: ", dimension(b))
# [END: init_basis]

# [ID: build_op]
ops = OpSum()
J = 1.0
for i in 0:8
    add_term!(ops, J, "Sz", i, "Sz", i + 1)
    add_term!(ops, J*0.5, "Sp", i, "Sm", i + 1)
    add_term!(ops, J*0.5, "Sm", i, "Sp", i + 1)
end
# [END: build_op]

# [ID: solve]
H = MatrixFreeHamiltonian(b, ops)
res = lanczos_ground_state(H)
println("Ground State Energy: ", res.energy)
# [END: solve]
