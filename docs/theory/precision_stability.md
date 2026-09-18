# Precision Stability: Ghost States and Orthogonality

Finite precision math cause "Loss of Orthogonality" in Lanczos. Krylov vectors lose independence. Lead to "Ghost States" (fake eigenvalues).

- **FP32:** Fast compute. Pollute subspace fast. Need more iterations.
- **FP64:** Slow memory read/write. Pristine subspace. Need less iterations.
