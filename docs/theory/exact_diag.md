# Exact Diagonalization (ED)

Exact Diagonalization (ED) is an unbiased, numerically exact method for solving quantum many-body lattice models without approximation or sign problems.

---

## 1. The Curse of Dimensionality

In quantum lattice systems with $N$ sites, each with a local Hilbert space of dimension $d$, the total Hilbert space dimension scales exponentially:

$$D = d^N$$

- **Spin-1/2 systems ($d=2$)**: $D = 2^N$. For $N=30$, $D = 2^{30} \approx 1.07 \times 10^9$.
- **Fermionic Hubbard models ($d=4$)**: $D = 4^N = 2^{2N}$. For $N=16$, $D \approx 4.29 \times 10^9$.

Storing a full dense Hamiltonian matrix requires $16 D^2$ bytes (for double-precision complex numbers). Even for a modest $N=16$ spin-1/2 system ($D = 65,536$), a dense matrix requires ~68.7 GB of RAM, while full diagonalization ($O(D^3)$) requires trillions of floating-point operations.

---

## 2. Symmetry Reduction & Block Diagonalization

Physical Hamiltonians often conserve total quantum numbers such as total spin projection $S^z$, particle number $N_p$, or spatial lattice translation momentum $k$:

$$[H, S^z] = 0, \quad [H, \hat{N}] = 0$$

Under these commuting symmetries, the Hamiltonian block-diagonalizes into independent invariant sectors:

$$H = \bigoplus_{s} H_s$$

For example, restricting an $N=20$ spin-1/2 Heisenberg chain to the zero-magnetization sector $S^z = 0$ reduces the dimension from $2^{20} = 1,048,576$ to:

$$D_{S^z=0} = \binom{20}{10} = 184,756$$

an over 5.6-fold reduction in state dimension.

---

## 3. Sparsity of Many-Body Hamiltonians

Typical physical interactions are local (e.g. nearest-neighbor exchange $J \sum_{\langle i, j \rangle} \vec{S}_i \cdot \vec{S}_j$). Consequently, each basis state $|n\rangle$ connects to only $\mathcal{O}(N)$ other basis states under $H$. 

The sparsity ratio is:

$$\frac{\text{Nonzeros}}{D^2} \approx \frac{z D}{D^2} = \frac{z}{D} \to 0 \quad \text{as } N \to \infty$$

where $z \propto N$ is the coordination number times number of interaction terms. In an $N=24$ system ($D \approx 2.7 \times 10^6$), more than $99.999\%$ of the Hamiltonian matrix entries are identically zero.

---

## 4. The Matrix-Free Paradigm

Traditional sparse solvers store matrix non-zero elements using formats such as Compressed Sparse Row (CSR). However, as $D$ reaches $10^7 - 10^9$, even storing column indices and values requires tens of gigabytes of RAM before performing any calculation.

`qkrylov` eliminates explicit matrix storage entirely:
- The operator expression is represented as an [`OpSum`](../models/index.md).
- The action $y = H x$ is evaluated directly by applying local operator transformations to state bitstrings on-the-fly.
- Memory consumption scales strictly as $\mathcal{O}(D)$ (the size of the vectors), allowing researchers to push exact diagonalization to the limits of physical hardware.
