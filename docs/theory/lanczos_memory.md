# The Memory Wall: Single-Pass vs Two-Pass Lanczos

Standard Lanczos (Single-Pass) store all Krylov vectors. Cost memory O(k*N) for k iterations. Hit memory wall fast for big L.

Two-Pass Lanczos save memory. First pass compute tridiagonal matrix T, only keep 3 vectors in memory. Cost O(N). Second pass recompute vectors to build ground state. Trade compute for memory, break memory wall.
