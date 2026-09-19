import time
import numpy as np
from qkrylov import SpinHalfBasis, SpinHalfSite, OpSum, MatrixFreeHamiltonian, lanczos_ground_state
from qkrylov.operators import Sz, Sp, Sm

import argparse

# ==============================================================================
# CONFIGURATION
# ==============================================================================
parser = argparse.ArgumentParser(description="QKrylov Multi-Hardware Benchmark")
parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Target device (cpu or cuda)")
parser.add_argument("--precision", type=str, default="32", choices=["32", "64"], help="Float precision (32 or 64)")
parser.add_argument("--L", type=int, default=4, help="System size (number of spins)")
args = parser.parse_args()

DEVICE = args.device
L = args.L
DTYPE = np.float64 if args.precision == "64" else np.float32
# ==============================================================================

print(f"--- QKrylov Multi-Hardware Benchmark ---")
print(f"Target Device: {DEVICE}")
print(f"Precision: FP{args.precision}")
print(f"System: 1D Heisenberg Model ({L} spins)\n")

# Build a 1D Heisenberg model OpSum
ops = OpSum(dtype=DTYPE)
for i in range(L - 1):
    ops += 1.0 * Sz(i) * Sz(i + 1)
    ops += 0.5 * Sp(i) * Sm(i + 1)
    ops += 0.5 * Sm(i) * Sp(i + 1)
# Periodic boundary condition
ops += 1.0 * Sz(L - 1) * Sz(0)
ops += 0.5 * Sp(L - 1) * Sm(0)
ops += 0.5 * Sm(L - 1) * Sp(0)

# 1. Build host-side structures
print("1. Building Basis and OpSum (Host CPU)...")
t0 = time.perf_counter()
basis = SpinHalfBasis(L, dtype=DTYPE)
site = SpinHalfSite(dtype=DTYPE)
t1 = time.perf_counter()
print(f"   -> Done in {t1 - t0:.4f} seconds.")
print(f"   -> Hilbert space dimension: {basis.size:,}")

# 2. Compile Hamiltonian to device (CSR generation)
print(f"\n2. Compiling Matrix-Free Hamiltonian to [{DEVICE}] with FP{args.precision}...")
t2 = time.perf_counter()
H = MatrixFreeHamiltonian(basis, site, ops, device=DEVICE, dtype=DTYPE)
t3 = time.perf_counter()
print(f"   -> Done in {t3 - t2:.4f} seconds.")

# 3. Run Lanczos Solver
print(f"\n3. Running Lanczos Solver on [{DEVICE}] (max 100 iterations)...")
t4 = time.perf_counter()
result = lanczos_ground_state(H, maxiter=100, tol=1e-8)
t5 = time.perf_counter()

print(f"   -> Done in {t5 - t4:.4f} seconds.")
print(f"\nResults:")
print(f"  Ground State Energy: {result.energy:.10f}")
print("\nBenchmark Complete.")
