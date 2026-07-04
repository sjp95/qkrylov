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
