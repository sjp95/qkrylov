# Ecosystem Integration

`qkrylov` play nice with ecosystem. After find ground state of 1D Heisenberg model, easy convert state vectors and operators.

## NumPy Integration
Convert solved eigenvector to NumPy array.

```python
import numpy as np
from qkrylov.solvers import Lanczos

# Solve H
solver = Lanczos(H)
E0, psi0 = solver.get_ground_state()

# Convert state to NumPy
psi_np = np.array(psi0)
```

## SciPy Sparse Matrix
Extract Hamiltonian as SciPy sparse matrix.

```python
import scipy.sparse as sp

# Get CSR sparse matrix
H_sparse = H.to_scipy_sparse(format="csr")
```

## PyTorch Integration
Move state vectors to PyTorch tensors for downstream pipelines.

```python
import torch

# Convert qkrylov state to torch.Tensor
psi_torch = torch.from_numpy(np.array(psi0)).to(device='cuda')
```
