# Physics Examples

## Heisenberg Antiferromagnet (2 sites)

```python
import qkrylov

basis = qkrylov.SpinHalfBasis(2)
site = qkrylov.SpinHalfSite()
os = qkrylov.OpSum()

# H = Sz0*Sz1 + 0.5(Sp0*Sm1 + Sm0*Sp1)
t1 = qkrylov.OperatorTerm()
t1.coeff = 1.0
t1.factors = [qkrylov.OperatorFactor("Sz", 0), qkrylov.OperatorFactor("Sz", 1)]
os.add_term(t1)

t2 = qkrylov.OperatorTerm()
t2.coeff = 0.5
t2.factors = [qkrylov.OperatorFactor("Sp", 0), qkrylov.OperatorFactor("Sm", 1)]
os.add_term(t2)

t3 = qkrylov.OperatorTerm()
t3.coeff = 0.5
t3.factors = [qkrylov.OperatorFactor("Sm", 0), qkrylov.OperatorFactor("Sp", 1)]
os.add_term(t3)

H = qkrylov.MatrixFreeHamiltonian(basis, site, os)
res = qkrylov.lanczos_ground_state(H)
print(f"Energy: {res.energy}") # Expected: -0.75
```

## Spinless Fermions on a Ring

```python
import qkrylov

N = 6
basis = qkrylov.FermionBasis(N)
site = qkrylov.FermionSite()
os = qkrylov.OpSum()

t = 1.0 # Hopping
for i in range(N):
    j = (i + 1) % N
    term = qkrylov.OperatorTerm()
    term.coeff = -t
    term.factors = [qkrylov.OperatorFactor("Cdag", i), qkrylov.OperatorFactor("C", j)]
    os.add_term(term)

    # H.c.
    term_hc = qkrylov.OperatorTerm()
    term_hc.coeff = -t
    term_hc.factors = [qkrylov.OperatorFactor("Cdag", j), qkrylov.OperatorFactor("C", i)]
    os.add_term(term_hc)

H = qkrylov.MatrixFreeHamiltonian(basis, site, os)
res = qkrylov.davidson_lowest(H, n_eig=1)
print(f"Ground state energy: {res.eigenvalues[0]}")
```

## 3-Spin Dzyaloshinskii–Moriya Interaction (DMI)

The 3-spin DMI interaction is given by $\vec{S}_i \cdot (\vec{S}_j \times \vec{S}_k)$. This can be expanded as:
$S_i^x(S_j^y S_k^z - S_j^z S_k^y) + S_i^y(S_j^z S_k^x - S_j^x S_k^z) + S_i^z(S_j^x S_k^y - S_j^y S_k^x)$

```python
import qkrylov

basis = qkrylov.SpinHalfBasis(3)
site = qkrylov.SpinHalfSite()
os = qkrylov.OpSum()

i, j, k = 0, 1, 2

# Terms for Sx_i (Sy_j Sz_k - Sz_j Sy_k)
# ... using Sx = 0.5(Sp+Sm), Sy = -0.5i(Sp-Sm) ...

# qkrylov handles complex coefficients and arbitrary operator strings automatically:
os += 0.25j, "Sp", i, "Sp", j, "Sz", k   # part of Sx_i Sy_j Sz_k
os += -0.25j, "Sp", i, "Sm", j, "Sz", k  # part of Sx_i Sy_j Sz_k
# ... and so on for all components of the cross product ...

H = qkrylov.MatrixFreeHamiltonian(basis, site, os)
res = qkrylov.lanczos_ground_state(H)
print(f"DMI Ground state energy: {res.energy}")
```

## Finite Temperature (FTLM) on Heisenberg Chain

```python
import qkrylov

N = 8
basis = qkrylov.SpinHalfBasis(N)
site = qkrylov.SpinHalfSite()
os = qkrylov.OpSum()

for i in range(N - 1):
    os += 1.0, "Sz", i, "Sz", i+1
    os += 0.5, "Sp", i, "Sm", i+1
    os += 0.5, "Sm", i, "Sp", i+1

H = qkrylov.MatrixFreeHamiltonian(basis, site, os)

betas = [0.1, 1.0, 10.0]
for beta in betas:
    res = qkrylov.ftlm(H, beta, n_random=100)
    print(f"Beta: {beta}, E: {res.internal_energy}, Cv: {res.specific_heat}")
```

## Mixed System: Fermion-Phonon Interaction (Holstein Model)

```python
import qkrylov

# 2 sites: each has a Fermion and a Phonon (Boson)
# We can use GenericBasis to define mixed local dimensions
# Site 0: Fermion (dim 2), Site 1: Phonon (dim 4), Site 2: Fermion (dim 2), Site 3: Phonon (dim 4)
local_dims = [2, 4, 2, 4]
basis = qkrylov.GenericBasis(local_dims)

sites = [
    qkrylov.FermionSite(),
    qkrylov.BosonSite(max_occupancy=3),
    qkrylov.FermionSite(),
    qkrylov.BosonSite(max_occupancy=3)
]

os = qkrylov.OpSum()
# Electron hopping between site 0 and 2
os += -1.0, "Cdag", 0, "C", 2
os += -1.0, "Cdag", 2, "C", 0

# Electron-phonon coupling (g * n_i * (b_i + bdag_i))
g = 0.5
os += g, "N", 0, "B", 1
os += g, "N", 0, "Bdag", 1
os += g, "N", 2, "B", 3
os += g, "N", 2, "Bdag", 3

# Phonon energy (omega * bdag * b)
omega_ph = 1.0
os += omega_ph, "N", 1
os += omega_ph, "N", 3

H = qkrylov.MatrixFreeHamiltonian(basis, sites, os)
res = qkrylov.lanczos_ground_state(H)
print(f"Holstein Ground state energy: {res.energy}")
```

## Mixed Spin System: Alternating Spin-1/2 and Spin-1

```python
import qkrylov

# 4 sites: alternating Spin-1/2 (dim 2) and Spin-1 (dim 3)
local_dims = [2, 3, 2, 3]
basis = qkrylov.GenericBasis(local_dims)

sites = [
    qkrylov.SpinHalfSite(),
    qkrylov.SpinSSite(S=1.0),
    qkrylov.SpinHalfSite(),
    qkrylov.SpinSSite(S=1.0)
]

os = qkrylov.OpSum()
# Nearest-neighbor Heisenberg exchange
for i in range(3):
    os += 1.0, "Sz", i, "Sz", i+1
    os += 0.5, "Sp", i, "Sm", i+1
    os += 0.5, "Sm", i, "Sp", i+1

H = qkrylov.MatrixFreeHamiltonian(basis, sites, os)
res = qkrylov.lanczos_ground_state(H)
print(f"Mixed Spin Ground state energy: {res.energy}")
```
