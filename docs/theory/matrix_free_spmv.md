# Matrix-Free SpMV in Quantum Many-Body Systems

Quantum systems have big Hilbert space (dim 2^L or N). Matrix H too big to store. SpMV (Sparse Matrix-Vector Multiply) need matrix-free way. Instead of store H, compute H|v> on fly. Use bitwise operations on basis states to find non-zero elements. Save memory, scale good. 
