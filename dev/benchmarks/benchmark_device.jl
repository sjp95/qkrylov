# Benchmark comparing repeated host-copy apply vs device-resident apply
using QuantumKrylov

function run_benchmark(L=14, n_rep=100)
    println("=== QKrylov GPU/Device Zero-Copy SpMV Benchmark ===")
    println("System: 1D Heisenberg Model (L = $L spins)")
    
    basis = SpinHalfBasis(L)
    dim = dimension(basis)
    println("Hilbert space dimension: $dim")
    
    ops = OpSum()
    for i in 0:(L-2)
        ops += 1.0 * Sz(i) * Sz(i+1)
        ops += 0.5 * Sp(i) * Sm(i+1)
        ops += 0.5 * Sm(i) * Sp(i+1)
    end
    ops += 1.0 * Sz(L-1) * Sz(0)
    ops += 0.5 * Sp(L-1) * Sm(0)
    ops += 0.5 * Sm(L-1) * Sp(0)
    
    H = MatrixFreeHamiltonian{Float64}(basis, ops)
    
    # Warmup
    x_host = rand(ComplexF64, dim)
    _ = H * x_host
    
    x_dev = DeviceVector(x_host)
    y_dev = DeviceVector{Float64}(dim)
    mul!(y_dev, H, x_dev)
    
    # 1. Host Staged Apply (copies host -> device -> kernel -> device -> host each iteration)
    t0 = time_ns()
    y_h = copy(x_host)
    for _ in 1:n_rep
        y_h = H * y_h
    end
    t_host = (time_ns() - t0) / 1e9
    
    # 2. Device-Resident Zero-Copy Apply (stays in device memory)
    t1 = time_ns()
    for _ in 1:n_rep
        mul!(y_dev, H, x_dev)
        vector_copy!(x_dev, y_dev)
    end
    t_dev = (time_ns() - t1) / 1e9
    
    println("Repetitions: $n_rep")
    println("Host-staged SpMV total time:     $(round(t_host, digits=4)) s ($(round(t_host / n_rep * 1000, digits=4)) ms/iter)")
    println("Device-resident SpMV total time:   $(round(t_dev, digits=4)) s ($(round(t_dev / n_rep * 1000, digits=4)) ms/iter)")
    speedup = t_host / t_dev
    println("Zero-copy staging speedup:       $(round(speedup, digits=2))x")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark(12, 100)
end
