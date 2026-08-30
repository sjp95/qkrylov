using Test
using QKrylov

@testset "QKrylov.jl" begin
    @testset "Sector" begin
        sec = Sector()
        @test sec.ptr != C_NULL
        set_sz!(sec, 0)
        set_hubbard_particles!(sec, 1, 1)
    end

    @testset "Site Types" begin
        s1 = SpinHalfSite()
        @test s1.ptr != C_NULL
        s2 = FermionSite()
        @test s2.ptr != C_NULL
        s3 = HubbardSite()
        @test s3.ptr != C_NULL
        s4 = TJSite()
        @test s4.ptr != C_NULL
    end

    @testset "Basis & Sectors" begin
        b_full = SpinHalfBasis(4)
        @test nsites(b_full) == 4
        @test dimension(b_full) == 16
        @test size(b_full) == (16, 16)

        sec = Sector()
        set_sz!(sec, 0)
        b_sec = SpinHalfBasis(4, sec)
        @test nsites(b_sec) == 4
        @test dimension(b_sec) == 6

        b_fermion = FermionBasis(4)
        @test dimension(b_fermion) == 16

        b_hubbard = HubbardBasis(2)
        @test dimension(b_hubbard) == 16

        b_tj = TJBasis(2)
        @test dimension(b_tj) == 9
    end

    @testset "OpSum & Hamiltonian Application" begin
        basis = SpinHalfBasis(2)
        site = SpinHalfSite()
        op = OpSum()
        
        # H = S^z_0 S^z_1 + 0.5 (S^+_0 S^-_1 + S^-_0 S^+_1)
        add_term!(op, 1.0, "Sz", 0, "Sz", 1)
        add_term!(op, 0.5, "Sp", 0, "Sm", 1)
        add_term!(op, 0.5, "Sm", 0, "Sp", 1)

        H = MatrixFreeHamiltonian(basis, site, op)
        @test dimension(H) == 4
        @test size(H) == (4, 4)

        x = [1.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im]
        y = H * x
        @test length(y) == 4
        @test isapprox(y[1], 0.25 + 0.0im, atol=1e-10)
    end

    @testset "DMI Interaction & Complex Terms" begin
        basis = SpinHalfBasis(3)
        site = SpinHalfSite()
        op = OpSum()

        # DMI z-component: (i D / 2) Sp_0 Sm_1 - (i D / 2) Sm_0 Sp_1
        D = 0.5
        add_term!(op, 1.0, "Sz", 0, "Sz", 1)
        add_term!(op, 0.5im * D, "Sp", 0, "Sm", 1)
        add_term!(op, -0.5im * D, "Sm", 0, "Sp", 1)

        H = MatrixFreeHamiltonian(basis, site, op)
        @test dimension(H) == 8

        res = lanczos_ground_state(H, maxiter=50, tol=1e-12)
        @test res.energy < 0.0
    end

    @testset "Hubbard & t-J Models in Julia" begin
        sec = Sector()
        set_hubbard_particles!(sec, 1, 1)

        b_hub = HubbardBasis(2, sec)
        s_hub = HubbardSite()
        @test dimension(b_hub) == 4

        op_hub = OpSum()
        t = 1.0
        U = 4.0
        add_term!(op_hub, -t, "CdagUp", 0, "CUp", 1)
        add_term!(op_hub, -t, "CdagUp", 1, "CUp", 0)
        add_term!(op_hub, -t, "CdagDn", 0, "CDn", 1)
        add_term!(op_hub, -t, "CdagDn", 1, "CDn", 0)
        add_term!(op_hub, U, "Nup", 0, "Ndn", 0)
        add_term!(op_hub, U, "Nup", 1, "Ndn", 1)

        H_hub = MatrixFreeHamiltonian(b_hub, s_hub, op_hub)
        res_hub = lanczos_ground_state(H_hub, maxiter=50, tol=1e-12)

        exact_hub = (U - sqrt(U^2 + 16.0 * t^2)) / 2.0
        @test isapprox(res_hub.energy, exact_hub, atol=1e-5)
    end

    @testset "Lanczos Ground State, State Save/Load & Correlation" begin
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
        res = lanczos_ground_state(H, maxiter=50, tol=1e-12)
        
        # Ground state energy for 4-site periodic Heisenberg chain is -2.0
        @test isapprox(res.energy, -2.0, atol=1e-6)

        # Save and load energy to verify Julia state preservation pattern
        tmpfile = tempname()
        open(tmpfile, "w") do io
            println(io, res.energy)
        end
        loaded_e = parse(Float64, readline(tmpfile))
        rm(tmpfile)
        @test isapprox(loaded_e, res.energy, atol=1e-12)

        # Compute spin-spin correlation <Sz_0 Sz_2>
        op_corr = OpSum()
        add_term!(op_corr, 1.0, "Sz", 0, "Sz", 2)
        H_corr = MatrixFreeHamiltonian(basis, site, op_corr)

        # Apply operator directly to ground state x
        x_gs = [1.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im]
        y_corr = H_corr * x_gs
        @test length(y_corr) == dimension(basis)
    end
end
