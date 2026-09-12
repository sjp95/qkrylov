/*
 * QKRYLOV FULLY ANNOTATED TUTORIAL
 */

/*
# Building the Hamiltonian

Welcome to the quickstart. In this tutorial, we will construct a 1D Heisenberg Spin-1/2 chain. 


## Step 1: The Hilbert Space

First, we define our Hilbert space. We use `SpinHalfBasis`.
*/

basis::SpinHalf b(10, sector::Sz(0));
    std::cout << "Basis dimension: " << b.dimension() << "\n";

/*
## Step 2: Operator Sum

Next, we use `OpSum` to symbolically build our interactions.
*/

OpSum ops;
    double J = 1.0;
    for (int i = 0; i < 9; ++i) {
        ops.add_term(OperatorTerm(J, {{"Sz", i}, {"Sz", i + 1}}));
        ops.add_term(OperatorTerm(J*0.5, {{"Sp", i}, {"Sm", i + 1}}));
        ops.add_term(OperatorTerm(J*0.5, {{"Sm", i}, {"Sp", i + 1}}));
    }

/*
## Step 3: Solve

Finally, we construct the matrix-free Hamiltonian and solve it.
*/

Hamiltonian H(b, ops);
    auto result = lanczos_ground_state(H);
    std::cout << "Ground State Energy: " << result.energy << "\n";
