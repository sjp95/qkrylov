using Test
using QuantumKrylov

@testset "QuantumKrylov.jl" begin
    @testset "Sector" begin
        sec = Sector()
        @test sec.ptr != C_NULL
        @test get_sz(sec) === nothing
        @test get_n(sec) === nothing
        @test string(sec) == "Sector(unconstrained)"

        set_sz!(sec, 0)
        @test get_sz(sec) == 0

        set_hubbard_particles!(sec, 1, 1)
        @test get_hubbard_particles(sec) == (1, 1)

        sec2 = Sector()
        set_n!(sec2, 2)
        @test get_n(sec2) == 2

        set_nb!(sec2, 1)
        @test get_nb(sec2) == 1
        @test string(sec2) == "Sector(N = 2, Nb = 1)"
    end

    @testset "Site Types" begin
        s1 = SpinHalfSite()
        @test s1.ptr != C_NULL
        @test string(s1) == "SpinHalfSite(dim = 2, states = [↑, ↓])"

        s2 = FermionSite()
        @test s2.ptr != C_NULL
        @test string(s2) == "FermionSite(dim = 2, states = [0, 1])"

        s3 = HubbardSite()
        @test s3.ptr != C_NULL
        @test string(s3) == "HubbardSite(dim = 4, states = [0, ↑, ↓, ↑↓])"

        s4 = TJSite()
        @test s4.ptr != C_NULL
        @test string(s4) == "TJSite(dim = 3, states = [0, ↑, ↓])"
    end

    @testset "Basis & Sectors & Lookups" begin
        b_full = SpinHalfBasis(4)
        @test nsites(b_full) == 4
        @test dimension(b_full) == 16
        @test size(b_full) == (16, 16)
        @test length(b_full) == 16
        @test string(b_full) == "SpinHalfBasis(sites = 4, dim = 16)"

        # State lookups
        st0 = state(b_full, 0)
        @test st0 == UInt64(0)
        @test b_full[1] == UInt64(0)
        @test basis_index(b_full, UInt64(0)) == 0
        @test UInt64(0) in b_full
        @test UInt64(15) in b_full

        sec = Sector()
        set_sz!(sec, 0)
        b_sec = SpinHalfBasis(4, sec)
        @test nsites(b_sec) == 4
        @test dimension(b_sec) == 6

        sec_f = Sector()
        set_n!(sec_f, 2)
        b_fermion = FermionBasis(4, sec_f)
        @test nsites(b_fermion) == 4
        @test dimension(b_fermion) == 6

        b_hubbard = HubbardBasis(2)
        @test dimension(b_hubbard) == 16

        b_tj = TJBasis(2)
        @test dimension(b_tj) == 9
    end

    @testset "OpSum, N-Body Terms & Hamiltonian Inspection" begin
        basis = SpinHalfBasis(3)
        site = SpinHalfSite()
        op = OpSum()

        # 1-body & 2-body terms
        add_term!(op, 1.0, "Sz", 0, "Sz", 1)
        # 3-body term: S^z_0 S^z_1 S^z_2
        add_term!(op, 0.5, ["Sz", "Sz", "Sz"], [0, 1, 2])

        H = MatrixFreeHamiltonian(basis, site, op)
        @test dimension(H) == 8

        # Test 2-argument convenience constructor MatrixFreeHamiltonian(basis, opsum)
        H_auto = MatrixFreeHamiltonian(basis, op)
        @test dimension(H_auto) == 8
        @test diagonal(H_auto) == diagonal(H)

        # Extract diagonal
        diag_val = diagonal(H)
        @test length(diag_val) == 8

        x = zeros(ComplexF64, 8)
        x[8] = 1.0 + 0.0im # |111> spin down state
        y = H * x
        # Diagonal term for |111>: 1.0*(0.25) + 0.5*(0.125) = 0.3125
        @test isapprox(y[8], 0.3125 + 0.0im, atol=1e-6)
    end

    @testset "Operator Generator Functions & Arithmetic Overloading" begin
        # Test 1.0 * Sz(0) * Sz(1) + 0.5 * (Sp(0)*Sm(1) + Sm(0)*Sp(1))
        N = 4
        basis = SpinHalfBasis(N)
        op = OpSum()

        for i in 0:(N-1)
            next_i = mod(i + 1, N)
            op += 1.0 * Sz(i) * Sz(next_i) + 0.5 * (Sp(i) * Sm(next_i) + Sm(i) * Sp(next_i))
        end

        H = MatrixFreeHamiltonian(basis, op)
        res = lanczos_ground_state(H, maxiter=50, tol=1e-12)
        @test isapprox(res.energy, -2.0, atol=1e-6)
    end

    @testset "OpSum Printing & Validation" begin
        op = OpSum()
        @test isempty(op)
        @test length(op) == 0
        @test string(op) == "OpSum(empty)"

        op += 1.0 * Sz(0) * Sz(1) + 0.5 * Sp(0) * Sm(1)
        @test !isempty(op)
        @test length(op) == 2
        @test string(op) == "OpSum(Sz(0) * Sz(1) + 0.5 * Sp(0) * Sm(1))"

        # Validation: valid vs out-of-bounds site
        @test validate(op, 4)[1] == true
        @test validate(op, 1)[1] == false # site 1 out of bounds for 1-site system

        # Pre-validation error when creating Hamiltonian with invalid site index
        bad_op = OpSum()
        bad_op += 1.0 * Sz(0) * Sz(10) # site 10 out of bounds for 4-site basis
        basis_4 = SpinHalfBasis(4)
        @test_throws ArgumentError MatrixFreeHamiltonian(basis_4, bad_op)
    end

    @testset "Lanczos Ground State Solver" begin
        # 4-site 1D Heisenberg chain
        N = 4
        basis = SpinHalfBasis(N)
        site = SpinHalfSite()
        op = OpSum()

        for i in 0:(N-1)
            next_i = mod(i + 1, N)
            add_term!(op, 1.0, "Sz", i, "Sz", next_i)
            add_term!(op, 0.5, "Sp", i, "Sm", next_i)
            add_term!(op, 0.5, "Sm", i, "Sp", next_i)
        end

        H = MatrixFreeHamiltonian(basis, site, op)

        # 1. Energy-only calculation (default)
        res = lanczos_ground_state(H, maxiter=50, tol=1e-12)
        @test isapprox(res.energy, -2.0, atol=1e-6)
        @test res.converged == true
        @test res.iterations > 0
        @test_throws ErrorException res.state
        @test_throws ErrorException res.eigenvector

        # 2. Maxiter hit test
        res_limited = lanczos_ground_state(H, maxiter=2, tol=1e-15)
        @test res_limited.converged == false
        @test res_limited.iterations == 2
        @test contains(string(res_limited), "WARNING: maxiter hit")

        # 3. State vector calculation
        res_state = lanczos_ground_state(H, maxiter=50, tol=1e-12, return_state=true)
        @test isapprox(res_state.energy, -2.0, atol=1e-6)
        psi = res_state.state
        @test length(psi) == 16
        @test res_state.eigenvector === psi

        # 4. Verify H * psi ≈ E0 * psi
        H_psi = H * psi
        @test isapprox(H_psi, res_state.energy .* psi, atol=1e-5)

        # 5. Destructuring test
        E0, psi_destruct = lanczos_ground_state(H, return_state=true)
        @test isapprox(E0, -2.0, atol=1e-6)
        @test length(psi_destruct) == 16
    end

    @testset "SciML Problem-Algorithm solve() Interface" begin
        # 4-site 1D Heisenberg chain (L=4 <= 15)
        N = 4
        basis = SpinHalfBasis(N)
        site = SpinHalfSite()
        op = OpSum()
        for i in 0:(N-1)
            next_i = mod(i + 1, N)
            add_term!(op, 1.0, "Sz", i, "Sz", next_i)
            add_term!(op, 0.5, "Sp", i, "Sm", next_i)
            add_term!(op, 0.5, "Sm", i, "Sp", next_i)
        end
        H = MatrixFreeHamiltonian(basis, site, op)

        prob = GroundStateProblem(H)
        @test prob isa AbstractQuantumProblem
        @test prob.H === H

        # 1. SinglePass Lanczos
        alg_sp = Lanczos(variation=SinglePass(), maxiter=50, tol=1e-12)
        @test alg_sp isa AbstractQuantumAlgorithm
        @test alg_sp.variation isa SinglePass
        sol_sp = solve(prob, alg_sp)
        @test sol_sp isa AbstractQuantumSolution
        @test sol_sp isa GroundStateSolution
        @test isapprox(sol_sp.value, -2.0, atol=1e-6)
        @test isapprox(sol_sp.energy, -2.0, atol=1e-6)
        @test sol_sp.converged == true
        @test sol_sp.iterations > 0
        @test length(sol_sp.u) == 16
        @test length(sol_sp.state) == 16
        @test length(sol_sp.eigenvector) == 16
        @test sol_sp.u === sol_sp.eigenvector
        @test isapprox(H * sol_sp.u, sol_sp.value .* sol_sp.u, atol=1e-5)

        # Destructuring test
        E0, psi = sol_sp
        @test isapprox(E0, -2.0, atol=1e-6)
        @test length(psi) == 16
        @test psi === sol_sp.u

        # Default algorithm dispatch
        sol_default = solve(prob)
        @test isapprox(sol_default.value, -2.0, atol=1e-6)
        @test isapprox(sol_default.u, sol_sp.u, atol=1e-5)

        # 2. TwoPass Lanczos Variation
        alg_tp = Lanczos(variation=TwoPass(), maxiter=50, tol=1e-12, return_state=true)
        @test alg_tp.variation isa TwoPass
        sol_tp = solve(prob, alg_tp)
        @test isapprox(sol_tp.value, -2.0, atol=1e-6)
        @test isapprox(sol_tp.value, sol_sp.value, atol=1e-10)
        @test sol_tp.converged == true
        @test length(sol_tp.u) == 16
        @test isapprox(H * sol_tp.u, sol_tp.value .* sol_tp.u, atol=1e-5)

        # Destructuring TwoPass
        E0_tp, psi_tp = sol_tp
        @test isapprox(E0_tp, -2.0, atol=1e-6)
        @test length(psi_tp) == 16

        # 3. Energy-only calculations (return_state=false)
        sol_sp_no_state = solve(prob, Lanczos(variation=SinglePass(), maxiter=50, tol=1e-12, return_state=false))
        @test isapprox(sol_sp_no_state.value, -2.0, atol=1e-6)
        @test_throws ErrorException sol_sp_no_state.u
        @test_throws ErrorException sol_sp_no_state.state

        sol_tp_no_state = solve(prob, Lanczos(variation=TwoPass(), maxiter=50, tol=1e-12, return_state=false))
        @test isapprox(sol_tp_no_state.value, -2.0, atol=1e-6)
        @test_throws ErrorException sol_tp_no_state.u

        # 4. ExcitedStatesProblem with Davidson and Lanczos
        ex_prob = ExcitedStatesProblem(H, 2)
        @test ex_prob isa AbstractQuantumProblem
        sol_dav = solve(ex_prob, Davidson(n_eig=2, max_subspace=10, tol=1e-6))
        @test length(sol_dav.eigenvalues) == 2
        @test isapprox(sol_dav.eigenvalues[1], -2.0, atol=1e-5)

        sol_lanczos_ex = solve(ex_prob, Lanczos(maxiter=50, tol=1e-8))
        @test length(sol_lanczos_ex.eigenvalues) == 2
        @test isapprox(sol_lanczos_ex.eigenvalues[1], -2.0, atol=1e-5)
        @test isapprox(sol_lanczos_ex.eigenvalues[2], -1.0, atol=1e-5)
        @test length(sol_lanczos_ex.eigenvectors) == 2
        @test isapprox(sol_lanczos_ex.energy, -2.0, atol=1e-5)

        # Direct lanczos_lowest with warm-starting
        lowest_direct = lanczos_lowest(H; n_eig=2, maxiter=50, tol=1e-8, initial_vector=psi)
        @test length(lowest_direct.eigenvalues) == 2
        @test isapprox(lowest_direct.eigenvalues[1], -2.0, atol=1e-5)

        # 5. ThermalProblem with FTLM
        th_prob = ThermalProblem(H, 1.0)
        @test th_prob isa AbstractQuantumProblem
        sol_ftlm = solve(th_prob, FTLM(beta=1.0, n_random=5, n_steps=20))
        @test sol_ftlm.partition_function > 0.0

        th_sweep_prob = ThermalProblem(H, [0.5, 1.0]; observables=[H])
        sol_sweep = solve(th_sweep_prob, FTLM(n_random=5, n_steps=20))
        @test sol_sweep isa FTLMSweepResult
        @test length(sol_sweep.beta_grid) == 2
        @test isapprox(sol_sweep.observable_expectations[1][1], sol_sweep.internal_energies[1], rtol=1e-4)

        # 6. DynamicsProblem with ContinuedFraction
        dyn_prob = DynamicsProblem(H, psi)
        @test dyn_prob isa AbstractQuantumProblem
        sol_dyn = solve(dyn_prob, ContinuedFraction(n_iter=10))
        @test length(sol_dyn.alphas) > 0

        # 7. SpectralProblem with CorrectionVector
        spec_prob = SpectralProblem(H, psi, -2.0, 0.5, 0.1)
        @test spec_prob isa AbstractQuantumProblem
        sol_spec = solve(spec_prob, CorrectionVector(e0=-2.0, omega=0.5, eta=0.1, maxiter=50, tol=1e-6))
        @test sol_spec.spectral_function >= 0.0
    end

    @testset "Davidson Solver" begin
        N = 4
        basis = SpinHalfBasis(N)
        site = SpinHalfSite()
        op = OpSum()
        for i in 0:(N-1)
            next_i = mod(i + 1, N)
            add_term!(op, 1.0, "Sz", i, "Sz", next_i)
            add_term!(op, 0.5, "Sp", i, "Sm", next_i)
            add_term!(op, 0.5, "Sm", i, "Sp", next_i)
        end
        H = MatrixFreeHamiltonian(basis, site, op)

        res = davidson_lowest(H, n_eig=2, max_subspace=10, tol=1e-6)
        @test length(res.eigenvalues) == 2
        @test isapprox(res.eigenvalues[1], -2.0, atol=1e-5)
        @test res.eigenvectors !== nothing
        @test length(res.eigenvectors) == 2
        @test length(res.eigenvectors[1]) == 16
    end

    @testset "Dynamics & Spectral Function" begin
        N = 4
        basis = SpinHalfBasis(N)
        site = SpinHalfSite()
        op = OpSum()
        for i in 0:(N-1)
            next_i = mod(i + 1, N)
            add_term!(op, 1.0, "Sz", i, "Sz", next_i)
            add_term!(op, 0.5, "Sp", i, "Sm", next_i)
            add_term!(op, 0.5, "Sm", i, "Sp", next_i)
        end
        H = MatrixFreeHamiltonian(basis, site, op)

        phi0 = zeros(ComplexF64, 16)
        phi0[1] = 1.0

        cfr = continued_fraction_coeffs(H, phi0, n_iter=10)
        @test length(cfr.alphas) > 0
        @test length(cfr.betas) >= 0

        spec_val = evaluate_spectral_function(cfr, 0.5, -2.0, 0.1)
        @test spec_val >= 0.0
    end

    @testset "FTLM Solver" begin
        N = 4
        basis = SpinHalfBasis(N)
        site = SpinHalfSite()
        op = OpSum()
        for i in 0:(N-1)
            next_i = mod(i + 1, N)
            add_term!(op, 1.0, "Sz", i, "Sz", next_i)
            add_term!(op, 0.5, "Sp", i, "Sm", next_i)
            add_term!(op, 0.5, "Sm", i, "Sp", next_i)
        end
        H = MatrixFreeHamiltonian(basis, site, op)

        ftlm_res = ftlm(H, beta=1.0, n_random=5, n_steps=20)
        @test isapprox(ftlm_res.beta, 1.0)
        @test ftlm_res.partition_function > 0.0

        sweep_res = ftlm_sweep(H; betas=[0.5, 1.0, 2.0], observables=[H], n_random=10, n_steps=20, seed=123)
        @test sweep_res isa FTLMSweepResult
        @test length(sweep_res.beta_grid) == 3
        @test all(sweep_res.partition_functions .> 0.0)
        @test all(sweep_res.specific_heats .>= -1e-12)
        @test all(sweep_res.entropies .>= -1e-12)
        @test length(sweep_res.observable_expectations) == 1
        @test isapprox(sweep_res.observable_expectations[1], sweep_res.internal_energies, rtol=1e-4)
        @test length(sweep_res.observable_errors) == 1
        @test all(sweep_res.observable_errors[1] .>= 0.0)
    end

    @testset "Device & Hardware Query API" begin
        gpu_build = is_gpu_build()
        @test isa(gpu_build, Bool)
        @test gpu_build == false # Testing on CPU environment

        gpu_name = find_gpu()
        @test gpu_name === nothing

        gpus = gpu_count()
        @test isa(gpus, Int)
        @test gpus == 0

        @test initialize_device!("cpu") === nothing

        # Device keyword in Hamiltonian
        basis = SpinHalfBasis(2)
        op = OpSum()
        op += 1.0 * Sz(0) * Sz(1)

        H_cpu = MatrixFreeHamiltonian(basis, op; device="cpu")
        @test dimension(H_cpu) == 4
        @test H_cpu.device == "cpu"

        # Typed device traits
        H_cpu_trait = MatrixFreeHamiltonian(basis, op; device=CPUDevice())
        @test H_cpu_trait.device isa AbstractDevice
        @test H_cpu_trait.device isa CPUDevice
        @test H_cpu_trait.device == "cpu"
        @test H_cpu_trait.device == CPUDevice()

        # Requesting GPU on CPU build throws an informative ArgumentError
        @test_throws ArgumentError MatrixFreeHamiltonian(basis, op; device="cuda")
        @test_throws ArgumentError MatrixFreeHamiltonian(basis, op; device="gpu")
        @test_throws ArgumentError MatrixFreeHamiltonian(basis, op; device=CUDADevice())
        @test_throws ArgumentError MatrixFreeHamiltonian(basis, op; device=HIPDevice())
        @test_throws ArgumentError MatrixFreeHamiltonian(basis, op; device=SYCLDevice())
    end

    @testset "Hubbard & Boson Operator Generators" begin
        op_hub = OpSum()
        op_hub += 1.0 * CdagUp(0) * CUp(1) + 1.0 * CdagDn(0) * CDn(1)
        op_hub += 2.0 * Nup(0) + 2.0 * Ndn(0) + 4.0 * Nupdn(0)
        @test length(op_hub) == 5

        op_boson = OpSum()
        op_boson += 1.0 * Bdag(0) * B(1) + 0.5 * N(0)
        @test length(op_boson) == 2
    end

    @testset "Keyword Basis Constructors" begin
        # SpinHalfBasis keyword constructor
        b_sz = SpinHalfBasis(4; sz=0)
        @test dimension(b_sz) == 6
        @test nsites(b_sz) == 4

        b_unconstrained = SpinHalfBasis(4; sz=nothing)
        @test dimension(b_unconstrained) == 16

        # FermionBasis keyword constructor
        b_n = FermionBasis(4; n=2)
        @test dimension(b_n) == 6

        # HubbardBasis keyword constructor
        b_hub = HubbardBasis(2; nup=1, ndn=1)
        @test dimension(b_hub) == 4 # 2 choose 1 up * 2 choose 1 down = 4

        # TJBasis keyword constructor
        b_tj = TJBasis(2; nup=1, ndn=1)
        @test dimension(b_tj) == 2
    end

    @testset "Dual-Precision Pipeline (Float64 and Float32)" begin
        N = 4
        basis = SpinHalfBasis(N; sz=0)
        op = OpSum()
        for i in 0:(N-2)
            op += 1.0 * Sz(i) * Sz(i+1) + 0.5 * (Sp(i)*Sm(i+1) + Sm(i)*Sp(i+1))
        end

        # 1. FP64 (Default on CPU)
        H64 = MatrixFreeHamiltonian(basis, op)
        @test H64.precision === Float64
        @test dimension(H64) == 6

        d64 = diagonal(H64)
        @test eltype(d64) === Float64
        @test length(d64) == 6

        x64 = randn(ComplexF64, 6)
        x64 ./= sqrt(sum(abs2, x64))
        y64 = H64 * x64
        @test eltype(y64) === ComplexF64

        res64 = lanczos_ground_state(H64, maxiter=50, tol=1e-12, return_state=true)
        @test res64.energy isa Float64
        @test isapprox(res64.energy, -1.6160254037844386, atol=1e-10)
        @test res64.converged == true
        @test eltype(res64.state) === ComplexF64

        # 2. FP32 Explicit
        H32 = MatrixFreeHamiltonian(basis, op; precision=Float32)
        @test H32.precision === Float32
        @test dimension(H32) == 6

        d32 = diagonal(H32)
        @test eltype(d32) === Float32
        @test length(d32) == 6
        @test isapprox(Vector{Float64}(d32), d64, atol=1e-5)

        x32 = Vector{ComplexF32}(x64)
        y32 = H32 * x32
        @test eltype(y32) === ComplexF32
        @test isapprox(Vector{ComplexF64}(y32), y64, atol=1e-5)

        res32 = lanczos_ground_state(H32, maxiter=50, tol=1e-6, return_state=true)
        @test res32.energy isa Float32
        @test isapprox(res32.energy, Float32(-1.6160254), atol=1e-4)
        @test res32.converged == true || res32.iterations == dimension(H32)
        @test eltype(res32.state) === ComplexF32
        # 3. Solvers with FP64 & FP32
        dav64 = davidson_lowest(H64, n_eig=2, max_subspace=10, tol=1e-10)
        @test eltype(dav64.eigenvalues) === Float64
        @test isapprox(dav64.eigenvalues[1], res64.energy, atol=1e-10)

        dav32 = davidson_lowest(H32, n_eig=2, max_subspace=10, tol=1e-5)
        @test eltype(dav32.eigenvalues) === Float32
        @test isapprox(dav32.eigenvalues[1], res32.energy, atol=1e-4)
    end

    @testset "Spin-1 & Correction Vector Spectroscopy" begin
        # 4-site Spin-1 chain (S=1.0) with Sz=0 sector (dim = 19)
        N = 4
        site = SpinSSite(1.0)
        @test string(site) == "SpinSSite(S = 1.0, dim = 3)"

        basis = SpinSBasis(N, 1.0; sz=0)
        @test dimension(basis) == 19
        @test nsites(basis) == 4

        op = OpSum()
        for i in 0:(N-2)
            op += 1.0 * Sz(i) * Sz(i+1) + 0.5 * (Sp(i)*Sm(i+1) + Sm(i)*Sp(i+1))
        end

        H = MatrixFreeHamiltonian(basis, site, op)
        @test dimension(H) == 19
        @test H.precision === Float64

        # Ground state
        res_gs = lanczos_ground_state(H, maxiter=100, tol=1e-10, return_state=true)
        @test isapprox(res_gs.energy, -4.64575, atol=1e-4)
        @test res_gs.converged == true

        # Local operator O = Sz(0)
        op_sz0 = OpSum()
        op_sz0 += 1.0 * Sz(0)
        H_sz0 = MatrixFreeHamiltonian(basis, site, op_sz0)
        op_psi0 = H_sz0 * res_gs.state

        # Correction vector solver
        cv_res = solver_correction_vector(
            H, op_psi0;
            e0=res_gs.energy,
            omega=1.5,
            eta=0.1,
            maxiter=100,
            tol=1e-8,
            return_vector=true
        )
        @test cv_res.converged == true
        @test isapprox(cv_res.spectral_function, 0.032465, atol=1e-4)
        @test length(cv_res.vector) == 19
        @test eltype(cv_res.vector) === ComplexF64
    end

    @testset "Error Diagnostics & Message Retrieval" begin
        clear_last_error()
        @test get_last_error_message() == ""

        # Test invalid argument via C API
        status = ccall((:qkrylov_sector_set_sz, QuantumKrylov.libqkrylov), Cint, (Ptr{Cvoid}, Cint), C_NULL, Cint(0))
        @test status == QuantumKrylov.QKRYLOV_ERROR_INVALID_ARG
        err = get_last_error_message()
        @test !isempty(err)
        @test occursin("sector handle is null", err)

        # Test clearing
        clear_last_error()
        @test get_last_error_message() == ""

        # Test solver invalid maxiter throwing error containing diagnostic
        H = MatrixFreeHamiltonian(SpinHalfBasis(2), OpSum())
        err_thrown = try
            lanczos_ground_state(H; maxiter=-5)
            nothing
        catch e
            e
        end
        @test err_thrown isa ErrorException
        @test occursin("maxiter must be positive", err_thrown.msg)
    end

    @testset "Basis, Site & OpSum Reflection + Site::apply" begin
        # 1. Site Reflection & Action Evaluation
        sh_site = SpinHalfSite()
        @test site_type(sh_site) == :SpinHalf
        @test isapprox(spin(sh_site), 0.5)
        @test dimension_per_site(sh_site) == 2

        # apply Sz on site 0, state 1 (|1> = spin up)
        act = apply(sh_site, "Sz", 0, UInt64(1))
        @test act.valid == true
        @test act.new_state == 1
        @test isapprox(act.matrix_element, 0.5 + 0.0im)

        # apply Sz on site 0, state 0 (|0> = spin down)
        act = apply(sh_site, "Sz", 0, UInt64(0))
        @test act.valid == true
        @test act.new_state == 0
        @test isapprox(act.matrix_element, -0.5 + 0.0im)

        # apply Sp on site 0, state 0 -> state 1
        act = apply(sh_site, "Sp", 0, UInt64(0))
        @test act.valid == true
        @test act.new_state == 1
        @test isapprox(act.matrix_element, 1.0 + 0.0im)

        # apply Sp on site 0, state 1 -> annihilated
        act = apply(sh_site, "Sp", 0, UInt64(1))
        @test act.valid == false

        # apply Sx and Sy
        act_x = apply(sh_site, "Sx", 0, UInt64(0))
        @test act_x.valid == true && act_x.new_state == 1 && isapprox(act_x.matrix_element, 0.5)

        act_y = apply(sh_site, "Sy", 0, UInt64(0))
        @test act_y.valid == true && act_y.new_state == 1 && isapprox(act_y.matrix_element, 0.5im)

        # Invalid operator throws ErrorException with diagnostic
        @test_throws ErrorException apply(sh_site, "NONEXISTENT", 0, UInt64(0))

        # SpinSSite (S=1)
        s1_site = SpinSSite(1.0)
        @test site_type(s1_site) == :SpinS
        @test isapprox(spin(s1_site), 1.0)
        @test dimension_per_site(s1_site) == 3
        # S=1: state 0 (m_z=-1) + Sp -> state 1 (m_z=0), matrix element = sqrt(2)
        act_s1 = apply(s1_site, "Sp", 0, UInt64(0))
        @test act_s1.valid == true
        @test act_s1.new_state == 1
        @test isapprox(act_s1.matrix_element, sqrt(2.0) + 0.0im)

        # FermionSite
        ferm_site = FermionSite()
        @test site_type(ferm_site) == :Fermion
        @test dimension_per_site(ferm_site) == 2
        act_c = apply(ferm_site, "C", 0, UInt64(1))
        @test act_c.valid == true && act_c.new_state == 0 && isapprox(act_c.matrix_element, 1.0)

        # 2. Basis Reflection
        b_sh = SpinHalfBasis(4, sz=0.0)
        @test basis_type(b_sh) == :SpinHalf
        @test isapprox(spin(b_sh), 0.5)
        @test dimension_per_site(b_sh) == 2
        sec_b = sector(b_sh)
        @test sec_b !== nothing
        @test get_sz(sec_b) == 0

        b_s1 = SpinSBasis(2, 1.0)
        @test basis_type(b_s1) == :SpinS
        @test isapprox(spin(b_s1), 1.0)
        @test dimension_per_site(b_s1) == 3

        b_ferm = FermionBasis(3)
        @test basis_type(b_ferm) == :Fermion
        @test dimension_per_site(b_ferm) == 2

        # 3. OpSum Reflection
        op = OpSum()
        op += 2.5 * Sz(0)
        op += (-1.0 + 0.5im) * Sp(0) * Sm(1)
        @test length(op) == 2
        @test opsum_size(op) == 2
        @test size(op) == (2,)

        # Term 0 inspection
        c0, nfac0 = opsum_get_term_info(op, 0)
        @test isapprox(c0, 2.5 + 0.0im)
        @test nfac0 == 1
        name0, site0 = opsum_get_factor(op, 0, 0)
        @test name0 == "Sz"
        @test site0 == 0

        # Term 1 inspection
        c1, nfac1 = opsum_get_term_info(op, 1)
        @test isapprox(c1, -1.0 + 0.5im)
        @test nfac1 == 2
        name1_0, site1_0 = opsum_get_factor(op, 1, 0)
        name1_1, site1_1 = opsum_get_factor(op, 1, 1)
        @test name1_0 == "Sp" && site1_0 == 0
        @test name1_1 == "Sm" && site1_1 == 1

        # Indexing
        @test op[1].coeff == 2.5 + 0.0im
        @test op[2].factors == [("Sp", 0), ("Sm", 1)]
    end

    @testset "Kokkos Parallel Vector Operations" begin
        # 1. Dot Product (Float64 & Float32)
        x64 = ComplexF64[1+2im, 3+4im, -1, -2im, 2+1im]
        y64 = ComplexF64[2-1im, 3im, 4+1im, 1-1im, -2]
        @test isapprox(vector_dot(x64, y64), 6.0 + 7.0im)
        @test vector_dot(ComplexF64[], ComplexF64[]) == 0.0 + 0.0im
        @test_throws DimensionMismatch vector_dot(x64, y64[1:3])

        x32 = ComplexF32.(x64)
        y32 = ComplexF32.(y64)
        @test isapprox(vector_dot(x32, y32), 6.0f0 + 7.0f0im)

        # 2. Norm (Float64 & Float32)
        @test isapprox(vector_norm(ComplexF64[3+4im, 0]), 5.0)
        @test isapprox(vector_norm(ComplexF32[3+4im, 0]), 5.0f0)
        @test vector_norm(ComplexF64[]) == 0.0

        # 3. AXPY (Float64 & Float32)
        ax = ComplexF64[1+1im, 2]
        ay = ComplexF64[3-1im, 1+2im]
        vector_axpy!(2.0, ax, ay)
        @test isapprox(ay, ComplexF64[5+1im, 5+2im])
        @test_throws DimensionMismatch vector_axpy!(2.0, ax, ay[1:1])

        ax32 = ComplexF32[1+1im, 2]
        ay32 = ComplexF32[3-1im, 1+2im]
        vector_axpy!(2.0f0, ax32, ay32)
        @test isapprox(ay32, ComplexF32[5+1im, 5+2im])

        # 4. SCAL (Float64 & Float32)
        sx64 = ComplexF64[2+3im, -1+4im]
        vector_scal!(2im, sx64)
        @test isapprox(sx64, ComplexF64[-6+4im, -8-2im])

        sx32 = ComplexF32[2+3im, -1+4im]
        vector_scal!(2.0f0im, sx32)
        @test isapprox(sx32, ComplexF32[-6+4im, -8-2im])

        # 5. Normalize (Float64 & Float32)
        nx = ComplexF64[3, 4]
        vector_normalize!(nx)
        @test isapprox(vector_norm(nx), 1.0)
        @test isapprox(nx, ComplexF64[0.6, 0.8])

        # 6. Zero Fill
        vector_zero_fill!(nx)
        @test all(nx .== 0.0 + 0.0im)

        # 7. Copy
        c_src = ComplexF64[1+2im, 3-4im]
        c_dst = zeros(ComplexF64, 2)
        vector_copy!(c_dst, c_src)
        @test c_dst == c_src
        @test_throws DimensionMismatch vector_copy!(c_dst, ComplexF64[1])
    end

    @testset "DeviceVector & Zero-Copy GPU SpMV" begin
        # 1. Allocation & Properties
        dv64 = DeviceVector{Float64}(4)
        @test length(dv64) == 4
        @test size(dv64) == (4,)
        @test eltype(dv64) == ComplexF64
        @test pointer(dv64) != C_NULL
        @test string(dv64) == "DeviceVector{Float64}(dim = 4)"

        dv32 = DeviceVector{Float32}(4)
        @test length(dv32) == 4
        @test eltype(dv32) == ComplexF32
        @test pointer(dv32) != C_NULL
        @test string(dv32) == "DeviceVector{Float32}(dim = 4)"

        @test_throws ArgumentError DeviceVector{Float64}(-1)

        # 2. Staging Copies (Host <-> Device)
        h64 = ComplexF64[1+0.5im, -2+1.5im, 0-1im, 3+2im]
        dv_from_host = DeviceVector(h64)
        @test length(dv_from_host) == 4
        @test eltype(dv_from_host) == ComplexF64
        @test Vector(dv_from_host) == h64

        h32 = ComplexF32[1+0.5im, -2+1.5im, 0-1im, 3+2im]
        dv_from_host32 = DeviceVector(h32)
        @test Vector(dv_from_host32) == h32

        copy_dst = zeros(ComplexF64, 4)
        copyto!(copy_dst, dv_from_host)
        @test copy_dst == h64
        @test_throws DimensionMismatch copyto!(zeros(ComplexF64, 2), dv_from_host)
        @test_throws DimensionMismatch copyto!(dv_from_host, zeros(ComplexF64, 2))

        # 3. MatrixFreeHamiltonian Zero-Copy SpMV
        b = SpinHalfBasis(2)
        ops = OpSum()
        ops += 1.0 * Sz(0) * Sz(1)
        ops += 0.5 * Sp(0) * Sm(1)
        ops += 0.5 * Sm(0) * Sp(1)

        H64 = MatrixFreeHamiltonian{Float64}(b, ops)
        H32 = MatrixFreeHamiltonian{Float32}(b, ops)

        # Host SpMV reference
        y_ref64 = H64 * h64
        y_ref32 = H32 * h32

        # Device SpMV (allocating *)
        y_dev64 = H64 * dv_from_host
        @test y_dev64 isa DeviceVector{Float64}
        @test isapprox(Vector(y_dev64), y_ref64)

        y_dev32 = H32 * dv_from_host32
        @test y_dev32 isa DeviceVector{Float32}
        @test isapprox(Vector(y_dev32), y_ref32)

        # Device SpMV (in-place mul!)
        y_dev_inplace64 = DeviceVector{Float64}(4)
        mul!(y_dev_inplace64, H64, dv_from_host)
        @test isapprox(Vector(y_dev_inplace64), y_ref64)

        y_dev_inplace32 = DeviceVector{Float32}(4)
        mul!(y_dev_inplace32, H32, dv_from_host32)
        @test isapprox(Vector(y_dev_inplace32), y_ref32)

        @test_throws DimensionMismatch H64 * DeviceVector{Float64}(8)
        @test_throws DimensionMismatch mul!(DeviceVector{Float64}(8), H64, dv_from_host)

        # 4. Device Diagonal
        d_dev64 = diagonal_device(H64)
        d_host64 = diagonal(H64)
        @test isapprox(Vector(d_dev64), ComplexF64.(d_host64))

        d_dev32 = diagonal_device(H32)
        d_host32 = diagonal(H32)
        @test isapprox(Vector(d_dev32), ComplexF32.(d_host32))

        # 5. Device BLAS-1 Kernels
        x_dev = DeviceVector(h64)
        y_dev = DeviceVector(y_ref64)

        # Dot
        d_res = vector_dot(x_dev, y_dev)
        @test isapprox(d_res, vector_dot(h64, y_ref64))
        @test_throws DimensionMismatch vector_dot(x_dev, DeviceVector{Float64}(2))

        # Norm
        n_res = vector_norm(x_dev)
        @test isapprox(n_res, vector_norm(h64))

        # AXPY
        vector_axpy!(2.0, x_dev, y_dev)
        expected_axpy = 2.0 .* h64 .+ y_ref64
        @test isapprox(Vector(y_dev), expected_axpy)

        # SCAL
        vector_scal!(0.5, y_dev)
        @test isapprox(Vector(y_dev), 0.5 .* expected_axpy)

        # Normalize
        vector_normalize!(y_dev)
        @test isapprox(vector_norm(y_dev), 1.0)

        # Copy
        y_clone = copy(y_dev)
        @test isapprox(vector_norm(y_clone), 1.0)
        @test Vector(y_clone) == Vector(y_dev)

        # Zero Fill
        vector_zero_fill!(y_clone)
        @test isapprox(vector_norm(y_clone), 0.0)
        @test all(Vector(y_clone) .== 0.0 + 0.0im)
    end
end


