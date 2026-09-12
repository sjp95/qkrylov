# Lanczos solver wrappers

# SciML Problem Types
abstract type AbstractQuantumProblem end

struct GroundStateProblem{HType} <: AbstractQuantumProblem
    H::HType
end

struct ExcitedStatesProblem{HType} <: AbstractQuantumProblem
    H::HType
    n_eig::Int
end
ExcitedStatesProblem(H; n_eig::Integer=1) = ExcitedStatesProblem(H, Int(n_eig))

struct ThermalProblem{HType, BType, OType} <: AbstractQuantumProblem
    H::HType
    betas::BType
    observables::OType
    is_sweep::Bool
end
ThermalProblem(H; beta::Real=1.0, observables=MatrixFreeHamiltonian[]) = ThermalProblem(H, [Float64(beta)], observables, false)
ThermalProblem(H, beta::Real; observables=MatrixFreeHamiltonian[]) = ThermalProblem(H, [Float64(beta)], observables, false)
ThermalProblem(H, betas::AbstractVector{<:Real}; observables=MatrixFreeHamiltonian[]) = ThermalProblem(H, Vector{Float64}(betas), observables, true)

function Base.getproperty(prob::ThermalProblem, s::Symbol)
    if s === :beta
        return getfield(prob, :betas)[1]
    else
        return getfield(prob, s)
    end
end

function Base.propertynames(prob::ThermalProblem, private::Bool=false)
    return private ? fieldnames(ThermalProblem) : (:H, :betas, :observables, :beta, :is_sweep)
end

struct DynamicsProblem{HType, VType} <: AbstractQuantumProblem
    H::HType
    phi0::VType
end

struct SpectralProblem{HType, VType} <: AbstractQuantumProblem
    H::HType
    op_psi0::VType
    e0::Float64
    omega::Float64
    eta::Float64
end

# Algorithm Types & Variations
abstract type AbstractQuantumAlgorithm end
abstract type AbstractLanczosVariation end

struct SinglePass <: AbstractLanczosVariation end
struct TwoPass   <: AbstractLanczosVariation end

struct Lanczos{V<:AbstractLanczosVariation} <: AbstractQuantumAlgorithm
    variation::V
    maxiter::Int
    tol::Float64
    return_state::Bool
end

function Lanczos(;
    variation::AbstractLanczosVariation = SinglePass(),
    maxiter::Integer = 200,
    tol::Real = 1e-12,
    return_state::Bool = true,
    compute_eigenvector::Bool = return_state
)
    return Lanczos(variation, Int(maxiter), Float64(tol), return_state || compute_eigenvector)
end

struct Davidson <: AbstractQuantumAlgorithm
    n_eig::Int
    max_subspace::Int
    tol::Float64
    compute_eigenvectors::Bool
end
Davidson(; n_eig::Integer=1, max_subspace::Integer=20, tol::Real=1e-8, compute_eigenvectors::Bool=true) =
    Davidson(Int(n_eig), Int(max_subspace), Float64(tol), compute_eigenvectors)

struct FTLM <: AbstractQuantumAlgorithm
    beta::Float64
    n_random::Int
    n_steps::Int
    seed::UInt64
end
FTLM(; beta::Real=1.0, n_random::Integer=50, n_steps::Integer=100, seed::Integer=42) =
    FTLM(Float64(beta), Int(n_random), Int(n_steps), UInt64(seed))
FTLM(beta::Real, n_random::Integer, n_steps::Integer) =
    FTLM(Float64(beta), Int(n_random), Int(n_steps), UInt64(42))

struct ContinuedFraction <: AbstractQuantumAlgorithm
    n_iter::Int
end
ContinuedFraction(; n_iter::Integer=100) = ContinuedFraction(Int(n_iter))

struct CorrectionVector <: AbstractQuantumAlgorithm
    e0::Float64
    omega::Float64
    eta::Float64
    maxiter::Int
    tol::Float64
    return_vector::Bool
end
CorrectionVector(; e0::Real=0.0, omega::Real=0.0, eta::Real=0.1, maxiter::Integer=100, tol::Real=1e-8, return_vector::Bool=false) =
    CorrectionVector(Float64(e0), Float64(omega), Float64(eta), Int(maxiter), Float64(tol), return_vector)

struct LanczosResultFP32C
    energy::Cfloat
    iterations::Cint
    converged::Cint
end

struct LanczosResultFP64C
    energy::Cdouble
    iterations::Cint
    converged::Cint
end

struct DavidsonResultC
    iterations::Cint
    converged::Cint
end

const LanczosLowestResultC = DavidsonResultC

struct FTLMResultFP32C
    beta::Cfloat
    partition_function::Cfloat
    internal_energy::Cfloat
    specific_heat::Cfloat
end

struct FTLMResultFP64C
    beta::Cdouble
    partition_function::Cdouble
    internal_energy::Cdouble
    specific_heat::Cdouble
end

struct FTLMSweepResultFP32C
    num_betas::Cint
    num_observables::Cint
    beta_grid::Ptr{Cfloat}
    partition_functions::Ptr{Cfloat}
    free_energies::Ptr{Cfloat}
    internal_energies::Ptr{Cfloat}
    specific_heats::Ptr{Cfloat}
    entropies::Ptr{Cfloat}
    observable_expectations::Ptr{Cfloat}
    observable_errors::Ptr{Cfloat}
end

struct FTLMSweepResultFP64C
    num_betas::Cint
    num_observables::Cint
    beta_grid::Ptr{Cdouble}
    partition_functions::Ptr{Cdouble}
    free_energies::Ptr{Cdouble}
    internal_energies::Ptr{Cdouble}
    specific_heats::Ptr{Cdouble}
    entropies::Ptr{Cdouble}
    observable_expectations::Ptr{Cdouble}
    observable_errors::Ptr{Cdouble}
end

struct CorrectionVectorResultFP32C
    spectral_function::Cfloat
    iterations::Cint
    converged::Cint
end

struct CorrectionVectorResultFP64C
    spectral_function::Cdouble
    iterations::Cint
    converged::Cint
end

# Solution Interface
abstract type AbstractQuantumSolution end

struct GroundStateSolution{T<:Real, V} <: AbstractQuantumSolution
    energy::T
    eigenvector::V
    iterations::Int
    converged::Bool
    has_state::Bool

    function GroundStateSolution(energy::T, eigenvector::V, iterations::Integer, converged::Bool) where {T<:Real, V}
        return new{T, V}(energy, eigenvector, Int(iterations), converged, eigenvector !== nothing)
    end
end

function GroundStateSolution(energy::T, iterations::Integer, converged::Bool, state::V=nothing) where {T<:Real, V}
    return GroundStateSolution(energy, state, iterations, converged)
end

const LanczosResult{T} = GroundStateSolution{T, Union{Vector{Complex{T}}, Nothing}}
function LanczosResult(energy::T, iterations::Integer, converged::Bool, state::Union{Vector{Complex{T}}, Nothing}=nothing) where {T<:Real}
    return GroundStateSolution(energy, state, iterations, converged)
end

function Base.getproperty(sol::GroundStateSolution, sym::Symbol)
    if sym === :value || sym === :energy
        return getfield(sol, :energy)
    elseif sym === :u || sym === :state || sym === :eigenvector || sym === :vector
        vec = getfield(sol, :eigenvector)
        if vec === nothing || !getfield(sol, :has_state)
            error("Ground state wavefunction was not computed. Pass `return_state=true` to compute the state vector.")
        end
        return vec
    end
    return getfield(sol, sym)
end

function Base.propertynames(sol::GroundStateSolution, private::Bool=false)
    return private ? fieldnames(GroundStateSolution) : (:energy, :eigenvector, :iterations, :converged, :u, :value, :state)
end

function Base.iterate(sol::GroundStateSolution, state=1)
    if state == 1
        return (sol.energy, 2)
    elseif state == 2
        if !sol.has_state || sol.eigenvector === nothing
            error("Ground state wavefunction was not computed. Pass `return_state=true` to compute the state vector.")
        end
        return (sol.eigenvector, 3)
    else
        return nothing
    end
end

function Base.show(io::IO, sol::GroundStateSolution{T}) where {T}
    status_str = sol.converged ? "converged = true" : "WARNING: maxiter hit without converging!"
    evec = getfield(sol, :eigenvector)
    vec_len = evec !== nothing ? length(evec) : 0
    state_str = getfield(sol, :has_state) ? ", state = Vector{Complex{$T}}(dim=$vec_len)" : ""
    print(io, "GroundStateSolution{$T}(energy = $(sol.energy), iterations = $(sol.iterations), $status_str$state_str)")
end

# Multiple Dispatch solve(prob, alg)
function solve(prob::GroundStateProblem{<:MatrixFreeHamiltonian{Float64}}, alg::Lanczos{SinglePass}; kwargs...)
    H = prob.H
    dim = Int(dimension(H))
    res_c = Ref{LanczosResultFP64C}(LanczosResultFP64C(0.0, 0, 0))

    if alg.return_state
        psi = Vector{ComplexF64}(undef, dim)
        GC.@preserve psi begin
            status = ccall(
                (:qkrylov_lanczos_ground_state_complex_fp64, libqkrylov),
                Cint,
                (Ptr{Cvoid}, Cint, Cdouble, Ref{LanczosResultFP64C}, Ptr{Cdouble}),
                H.ptr, Cint(alg.maxiter), Cdouble(alg.tol), res_c, pointer(psi)
            )
        end
        _check_status(status, "Lanczos single-pass ground state solver failed")
        return GroundStateSolution(res_c[].energy, psi, Int(res_c[].iterations), res_c[].converged != 0)
    else
        status = ccall(
            (:qkrylov_lanczos_ground_state_fp64, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Cint, Cdouble, Ref{LanczosResultFP64C}),
            H.ptr, Cint(alg.maxiter), Cdouble(alg.tol), res_c
        )
        _check_status(status, "Lanczos single-pass ground state solver failed")
        return GroundStateSolution(res_c[].energy, nothing, Int(res_c[].iterations), res_c[].converged != 0)
    end
end

function solve(prob::GroundStateProblem{<:MatrixFreeHamiltonian{Float32}}, alg::Lanczos{SinglePass}; kwargs...)
    H = prob.H
    dim = Int(dimension(H))
    res_c = Ref{LanczosResultFP32C}(LanczosResultFP32C(0.0f0, 0, 0))

    if alg.return_state
        psi = Vector{ComplexF32}(undef, dim)
        GC.@preserve psi begin
            status = ccall(
                (:qkrylov_lanczos_ground_state_complex_fp32, libqkrylov),
                Cint,
                (Ptr{Cvoid}, Cint, Cfloat, Ref{LanczosResultFP32C}, Ptr{Cfloat}),
                H.ptr, Cint(alg.maxiter), Cfloat(alg.tol), res_c, pointer(psi)
            )
        end
        _check_status(status, "Lanczos single-pass ground state solver failed")
        return GroundStateSolution(res_c[].energy, psi, Int(res_c[].iterations), res_c[].converged != 0)
    else
        status = ccall(
            (:qkrylov_lanczos_ground_state_fp32, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Cint, Cfloat, Ref{LanczosResultFP32C}),
            H.ptr, Cint(alg.maxiter), Cfloat(alg.tol), res_c
        )
        _check_status(status, "Lanczos single-pass ground state solver failed")
        return GroundStateSolution(res_c[].energy, nothing, Int(res_c[].iterations), res_c[].converged != 0)
    end
end

function solve(prob::GroundStateProblem{<:MatrixFreeHamiltonian{Float64}}, alg::Lanczos{TwoPass}; kwargs...)
    H = prob.H
    dim = Int(dimension(H))
    res_c = Ref{LanczosResultFP64C}(LanczosResultFP64C(0.0, 0, 0))

    if alg.return_state
        psi = Vector{ComplexF64}(undef, dim)
        GC.@preserve psi begin
            status = ccall(
                (:qkrylov_lanczos_two_pass_ground_state_complex_fp64, libqkrylov),
                Cint,
                (Ptr{Cvoid}, Cint, Cdouble, Ref{LanczosResultFP64C}, Ptr{Cdouble}),
                H.ptr, Cint(alg.maxiter), Cdouble(alg.tol), res_c, pointer(psi)
            )
        end
        _check_status(status, "Lanczos two-pass ground state solver failed")
        return GroundStateSolution(res_c[].energy, psi, Int(res_c[].iterations), res_c[].converged != 0)
    else
        status = ccall(
            (:qkrylov_lanczos_two_pass_ground_state_fp64, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Cint, Cdouble, Ref{LanczosResultFP64C}),
            H.ptr, Cint(alg.maxiter), Cdouble(alg.tol), res_c
        )
        _check_status(status, "Lanczos two-pass ground state solver failed")
        return GroundStateSolution(res_c[].energy, nothing, Int(res_c[].iterations), res_c[].converged != 0)
    end
end

function solve(prob::GroundStateProblem{<:MatrixFreeHamiltonian{Float32}}, alg::Lanczos{TwoPass}; kwargs...)
    H = prob.H
    dim = Int(dimension(H))
    res_c = Ref{LanczosResultFP32C}(LanczosResultFP32C(0.0f0, 0, 0))

    if alg.return_state
        psi = Vector{ComplexF32}(undef, dim)
        GC.@preserve psi begin
            status = ccall(
                (:qkrylov_lanczos_two_pass_ground_state_complex_fp32, libqkrylov),
                Cint,
                (Ptr{Cvoid}, Cint, Cfloat, Ref{LanczosResultFP32C}, Ptr{Cfloat}),
                H.ptr, Cint(alg.maxiter), Cfloat(alg.tol), res_c, pointer(psi)
            )
        end
        _check_status(status, "Lanczos two-pass ground state solver failed")
        return GroundStateSolution(res_c[].energy, psi, Int(res_c[].iterations), res_c[].converged != 0)
    else
        status = ccall(
            (:qkrylov_lanczos_two_pass_ground_state_fp32, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Cint, Cfloat, Ref{LanczosResultFP32C}),
            H.ptr, Cint(alg.maxiter), Cfloat(alg.tol), res_c
        )
        _check_status(status, "Lanczos two-pass ground state solver failed")
        return GroundStateSolution(res_c[].energy, nothing, Int(res_c[].iterations), res_c[].converged != 0)
    end
end

solve(prob::GroundStateProblem; kwargs...) = solve(prob, Lanczos(); kwargs...)

function solve(prob::ExcitedStatesProblem, alg::Lanczos; kwargs...)
    return lanczos_lowest(prob.H; n_eig=prob.n_eig, maxiter=alg.maxiter, tol=alg.tol, compute_eigenvectors=alg.return_state)
end

function solve(prob::ExcitedStatesProblem, alg::Davidson=Davidson(); kwargs...)
    return davidson_lowest(prob.H; n_eig=alg.n_eig, max_subspace=alg.max_subspace, tol=alg.tol, compute_eigenvectors=alg.compute_eigenvectors)
end

function solve(prob::ThermalProblem, alg::FTLM=FTLM(); kwargs...)
    if prob.is_sweep || length(prob.betas) > 1 || !isempty(prob.observables)
        return ftlm_sweep(prob.H; betas=prob.betas, observables=prob.observables, n_random=alg.n_random, n_steps=alg.n_steps, seed=alg.seed)
    else
        return ftlm(prob.H; beta=prob.beta, n_random=alg.n_random, n_steps=alg.n_steps)
    end
end

function solve(prob::DynamicsProblem, alg::ContinuedFraction=ContinuedFraction(); kwargs...)
    return continued_fraction_coeffs(prob.H, prob.phi0; n_iter=alg.n_iter)
end

function solve(prob::SpectralProblem, alg::CorrectionVector=CorrectionVector(); kwargs...)
    return solver_correction_vector(prob.H, prob.op_psi0; e0=prob.e0, omega=prob.omega, eta=prob.eta, maxiter=alg.maxiter, tol=alg.tol, return_vector=alg.return_vector)
end

function lanczos_ground_state(
    H::MatrixFreeHamiltonian;
    maxiter::Integer=100,
    tol::Real=(H.precision === Float32 ? 1e-6 : 1e-12),
    return_state::Bool=false,
    compute_eigenvector::Bool=return_state
)
    return solve(GroundStateProblem(H), Lanczos(variation=SinglePass(), maxiter=maxiter, tol=tol, return_state=return_state || compute_eigenvector))
end

# Excited States Eigensolver Solutions (Davidson and Lanczos Lowest)
struct ExcitedStatesSolution{T<:Real} <: AbstractQuantumSolution
    eigenvalues::Vector{T}
    eigenvectors::Union{Vector{Vector{Complex{T}}}, Nothing}
    iterations::Int
    converged::Bool
end

const DavidsonResult{T} = ExcitedStatesSolution{T}
const LanczosLowestResult{T} = ExcitedStatesSolution{T}

function Base.getproperty(res::ExcitedStatesSolution{T}, sym::Symbol) where {T}
    if sym === :values || sym === :evals
        return getfield(res, :eigenvalues)
    elseif sym === :energy
        evals = getfield(res, :eigenvalues)
        return isempty(evals) ? zero(T) : evals[1]
    elseif sym === :vectors || sym === :evecs
        vecs = getfield(res, :eigenvectors)
        if vecs === nothing
            error("Eigenvectors were not computed for this run. Pass `compute_eigenvectors=true` (or `return_state=true`).")
        end
        return vecs
    elseif sym === :state || sym === :u || sym === :eigenvector
        vecs = getfield(res, :eigenvectors)
        if vecs === nothing || isempty(vecs)
            error("Eigenvectors were not computed or empty.")
        end
        return vecs[1]
    end
    return getfield(res, sym)
end

function Base.propertynames(res::ExcitedStatesSolution, private::Bool=false)
    return private ? fieldnames(ExcitedStatesSolution) : (:eigenvalues, :eigenvectors, :iterations, :converged, :values, :evals, :vectors, :evecs, :energy, :state, :u, :eigenvector)
end

function davidson_lowest(
    H::MatrixFreeHamiltonian{Float64};
    n_eig::Integer=1,
    max_subspace::Integer=20,
    tol::Real=1e-8,
    compute_eigenvectors::Bool=true
)::DavidsonResult{Float64}
    dim = Int(dimension(H))
    evals = Vector{Float64}(undef, n_eig)
    evecs_flat = compute_eigenvectors ? Vector{ComplexF64}(undef, n_eig * dim) : ComplexF64[]
    dav_c = Ref{DavidsonResultC}(DavidsonResultC(0, 0))

    GC.@preserve evals evecs_flat begin
        evecs_ptr = compute_eigenvectors ? pointer(evecs_flat) : Ptr{ComplexF64}(C_NULL)
        status = ccall(
            (:qkrylov_davidson_lowest_complex_fp64, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Cint, Cint, Cdouble, Ptr{Cdouble}, Ptr{Cdouble}, Ref{DavidsonResultC}),
            H.ptr, Cint(n_eig), Cint(max_subspace), Cdouble(tol), pointer(evals), Ptr{Cdouble}(evecs_ptr), dav_c
        )
        _check_status(status, "Davidson solver failed")
    end

    iters = Int(dav_c[].iterations)
    conv  = dav_c[].converged != 0

    if !compute_eigenvectors
        return DavidsonResult{Float64}(evals, nothing, iters, conv)
    end

    evecs = Vector{Vector{ComplexF64}}(undef, n_eig)
    for idx in 1:n_eig
        evecs[idx] = evecs_flat[(idx-1)*dim + 1 : idx*dim]
    end
    return DavidsonResult{Float64}(evals, evecs, iters, conv)
end

function davidson_lowest(
    H::MatrixFreeHamiltonian{Float32};
    n_eig::Integer=1,
    max_subspace::Integer=20,
    tol::Real=1e-5,
    compute_eigenvectors::Bool=true
)::DavidsonResult{Float32}
    dim = Int(dimension(H))
    evals = Vector{Float32}(undef, n_eig)
    evecs_flat = compute_eigenvectors ? Vector{ComplexF32}(undef, n_eig * dim) : ComplexF32[]
    dav_c = Ref{DavidsonResultC}(DavidsonResultC(0, 0))

    GC.@preserve evals evecs_flat begin
        evecs_ptr = compute_eigenvectors ? pointer(evecs_flat) : Ptr{ComplexF32}(C_NULL)
        status = ccall(
            (:qkrylov_davidson_lowest_complex_fp32, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Cint, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ref{DavidsonResultC}),
            H.ptr, Cint(n_eig), Cint(max_subspace), Cfloat(tol), pointer(evals), Ptr{Cfloat}(evecs_ptr), dav_c
        )
        _check_status(status, "Davidson solver failed")
    end

    iters = Int(dav_c[].iterations)
    conv  = dav_c[].converged != 0

    if !compute_eigenvectors
        return DavidsonResult{Float32}(evals, nothing, iters, conv)
    end

    evecs = Vector{Vector{ComplexF32}}(undef, n_eig)
    for idx in 1:n_eig
        evecs[idx] = evecs_flat[(idx-1)*dim + 1 : idx*dim]
    end
    return DavidsonResult{Float32}(evals, evecs, iters, conv)
end

function lanczos_lowest(
    H::MatrixFreeHamiltonian{Float64};
    n_eig::Integer=1,
    maxiter::Integer=200,
    tol::Real=1e-8,
    compute_eigenvectors::Bool=true,
    initial_vector::Union{AbstractVector, Nothing}=nothing
)::LanczosLowestResult{Float64}
    dim = Int(dimension(H))
    if initial_vector !== nothing
        if length(initial_vector) != dim
            throw(DimensionMismatch("initial_vector length ($(length(initial_vector))) does not match Hamiltonian dimension ($dim)"))
        end
    end
    init_vec = initial_vector === nothing ? nothing : (initial_vector isa Vector{ComplexF64} ? initial_vector : Vector{ComplexF64}(initial_vector))
    init_ptr = init_vec === nothing ? Ptr{Cdouble}(C_NULL) : Ptr{Cdouble}(pointer(init_vec))

    evals = Vector{Float64}(undef, n_eig)
    evecs_flat = compute_eigenvectors ? Vector{ComplexF64}(undef, n_eig * dim) : ComplexF64[]
    res_c = Ref{LanczosLowestResultC}(LanczosLowestResultC(0, 0))

    GC.@preserve evals evecs_flat init_vec begin
        evecs_ptr = compute_eigenvectors ? pointer(evecs_flat) : Ptr{ComplexF64}(C_NULL)
        status = ccall(
            (:qkrylov_lanczos_lowest_complex_fp64, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Cint, Cint, Cdouble, Ptr{Cdouble}, Ptr{Cdouble}, Ref{LanczosLowestResultC}, Ptr{Cdouble}),
            H.ptr, Cint(n_eig), Cint(maxiter), Cdouble(tol), pointer(evals), Ptr{Cdouble}(evecs_ptr), res_c, init_ptr
        )
        _check_status(status, "Lanczos lowest solver failed")
    end

    iters = Int(res_c[].iterations)
    conv  = res_c[].converged != 0

    if !compute_eigenvectors
        return LanczosLowestResult{Float64}(evals, nothing, iters, conv)
    end

    evecs = Vector{Vector{ComplexF64}}(undef, n_eig)
    for idx in 1:n_eig
        evecs[idx] = evecs_flat[(idx-1)*dim + 1 : idx*dim]
    end
    return LanczosLowestResult{Float64}(evals, evecs, iters, conv)
end

function lanczos_lowest(
    H::MatrixFreeHamiltonian{Float32};
    n_eig::Integer=1,
    maxiter::Integer=200,
    tol::Real=1e-5,
    compute_eigenvectors::Bool=true,
    initial_vector::Union{AbstractVector, Nothing}=nothing
)::LanczosLowestResult{Float32}
    dim = Int(dimension(H))
    if initial_vector !== nothing
        if length(initial_vector) != dim
            throw(DimensionMismatch("initial_vector length ($(length(initial_vector))) does not match Hamiltonian dimension ($dim)"))
        end
    end
    init_vec = initial_vector === nothing ? nothing : (initial_vector isa Vector{ComplexF32} ? initial_vector : Vector{ComplexF32}(initial_vector))
    init_ptr = init_vec === nothing ? Ptr{Cfloat}(C_NULL) : Ptr{Cfloat}(pointer(init_vec))

    evals = Vector{Float32}(undef, n_eig)
    evecs_flat = compute_eigenvectors ? Vector{ComplexF32}(undef, n_eig * dim) : ComplexF32[]
    res_c = Ref{LanczosLowestResultC}(LanczosLowestResultC(0, 0))

    GC.@preserve evals evecs_flat init_vec begin
        evecs_ptr = compute_eigenvectors ? pointer(evecs_flat) : Ptr{ComplexF32}(C_NULL)
        status = ccall(
            (:qkrylov_lanczos_lowest_complex_fp32, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Cint, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ref{LanczosLowestResultC}, Ptr{Cfloat}),
            H.ptr, Cint(n_eig), Cint(maxiter), Cfloat(tol), pointer(evals), Ptr{Cfloat}(evecs_ptr), res_c, init_ptr
        )
        _check_status(status, "Lanczos lowest solver failed")
    end

    iters = Int(res_c[].iterations)
    conv  = res_c[].converged != 0

    if !compute_eigenvectors
        return LanczosLowestResult{Float32}(evals, nothing, iters, conv)
    end

    evecs = Vector{Vector{ComplexF32}}(undef, n_eig)
    for idx in 1:n_eig
        evecs[idx] = evecs_flat[(idx-1)*dim + 1 : idx*dim]
    end
    return LanczosLowestResult{Float32}(evals, evecs, iters, conv)
end

# Dynamics & Spectral Function
struct ContinuedFractionResult{T<:Real}
    alphas::Vector{T}
    betas::Vector{T}
    norm_phi0::T
end

function continued_fraction_coeffs(
    H::MatrixFreeHamiltonian{Float64},
    phi0::AbstractVector{<:Number};
    n_iter::Integer=100
)::ContinuedFractionResult{Float64}
    dim = Int(dimension(H))
    @assert length(phi0) == dim "Initial vector phi0 size $(length(phi0)) does not match Hamiltonian dimension $dim"

    phi0_c = (phi0 isa Vector{ComplexF64}) ? phi0 : Vector{ComplexF64}(phi0)
    alphas_buf = Vector{Float64}(undef, n_iter)
    betas_buf  = Vector{Float64}(undef, n_iter)
    norm_ref   = Ref{Cdouble}(0.0)
    num_coeffs = Ref{Cint}(0)

    GC.@preserve phi0_c alphas_buf betas_buf begin
        status = ccall(
            (:qkrylov_continued_fraction_coeffs_complex_fp64, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ref{Cdouble}, Ref{Cint}),
            H.ptr, pointer(phi0_c), Cint(n_iter), pointer(alphas_buf), pointer(betas_buf), norm_ref, num_coeffs
        )
        _check_status(status, "Continued fraction solver failed")
    end

    k = Int(num_coeffs[])
    alphas = alphas_buf[1:k]
    betas  = betas_buf[1:max(0, k - 1)]
    return ContinuedFractionResult{Float64}(alphas, betas, norm_ref[])
end

function continued_fraction_coeffs(
    H::MatrixFreeHamiltonian{Float32},
    phi0::AbstractVector{<:Number};
    n_iter::Integer=100
)::ContinuedFractionResult{Float32}
    dim = Int(dimension(H))
    @assert length(phi0) == dim "Initial vector phi0 size $(length(phi0)) does not match Hamiltonian dimension $dim"

    phi0_c = (phi0 isa Vector{ComplexF32}) ? phi0 : Vector{ComplexF32}(phi0)
    alphas_buf = Vector{Float32}(undef, n_iter)
    betas_buf  = Vector{Float32}(undef, n_iter)
    norm_ref   = Ref{Cfloat}(0.0f0)
    num_coeffs = Ref{Cint}(0)

    GC.@preserve phi0_c alphas_buf betas_buf begin
        status = ccall(
            (:qkrylov_continued_fraction_coeffs_complex_fp32, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ref{Cfloat}, Ref{Cint}),
            H.ptr, pointer(phi0_c), Cint(n_iter), pointer(alphas_buf), pointer(betas_buf), norm_ref, num_coeffs
        )
        _check_status(status, "Continued fraction solver failed")
    end

    k = Int(num_coeffs[])
    alphas = alphas_buf[1:k]
    betas  = betas_buf[1:max(0, k - 1)]
    return ContinuedFractionResult{Float32}(alphas, betas, norm_ref[])
end

function evaluate_spectral_function(
    cfr::ContinuedFractionResult{T},
    omega::Real,
    E0::Real,
    eta::Real
)::T where {T<:Real}
    return evaluate_spectral_function(cfr.alphas, cfr.betas, cfr.norm_phi0, omega, E0, eta)
end

function evaluate_spectral_function(
    alphas::AbstractVector{Float64},
    betas::AbstractVector{Float64},
    norm_phi0::Real,
    omega::Real,
    E0::Real,
    eta::Real
)::Float64
    alphas_buf = (alphas isa Vector{Float64}) ? alphas : Vector{Float64}(alphas)
    betas_buf  = (betas isa Vector{Float64}) ? betas : Vector{Float64}(betas)
    n = length(alphas_buf)

    GC.@preserve alphas_buf betas_buf begin
        val = ccall(
            (:qkrylov_evaluate_spectral_function_fp64, libqkrylov),
            Cdouble,
            (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Cdouble, Cdouble, Cdouble, Cdouble),
            pointer(alphas_buf), pointer(betas_buf), Csize_t(n),
            Cdouble(norm_phi0), Cdouble(omega), Cdouble(E0), Cdouble(eta)
        )
    end
    return val
end

function evaluate_spectral_function(
    alphas::AbstractVector{Float32},
    betas::AbstractVector{Float32},
    norm_phi0::Real,
    omega::Real,
    E0::Real,
    eta::Real
)::Float32
    alphas_buf = (alphas isa Vector{Float32}) ? alphas : Vector{Float32}(alphas)
    betas_buf  = (betas isa Vector{Float32}) ? betas : Vector{Float32}(betas)
    n = length(alphas_buf)

    GC.@preserve alphas_buf betas_buf begin
        val = ccall(
            (:qkrylov_evaluate_spectral_function_fp32, libqkrylov),
            Cfloat,
            (Ptr{Cfloat}, Ptr{Cfloat}, Csize_t, Cfloat, Cfloat, Cfloat, Cfloat),
            pointer(alphas_buf), pointer(betas_buf), Csize_t(n),
            Cfloat(norm_phi0), Cfloat(omega), Cfloat(E0), Cfloat(eta)
        )
    end
    return val
end

function evaluate_spectral_function(
    alphas::AbstractVector{<:Real},
    betas::AbstractVector{<:Real},
    norm_phi0::Real,
    omega::Real,
    E0::Real,
    eta::Real
)::Float64
    return evaluate_spectral_function(Vector{Float64}(alphas), Vector{Float64}(betas), Float64(norm_phi0), Float64(omega), Float64(E0), Float64(eta))
end

# FTLM (Finite Temperature Lanczos) Solver
struct FTLMResult{T<:Real}
    beta::T
    partition_function::T
    internal_energy::T
    specific_heat::T
end

function ftlm(
    H::MatrixFreeHamiltonian{Float64};
    beta::Real=1.0,
    n_random::Integer=10,
    n_steps::Integer=50
)::FTLMResult{Float64}
    res_c = Ref{FTLMResultFP64C}(FTLMResultFP64C(0.0, 0.0, 0.0, 0.0))
    status = ccall(
        (:qkrylov_ftlm_fp64, libqkrylov),
        Cint,
        (Ptr{Cvoid}, Cdouble, Cint, Cint, Ref{FTLMResultFP64C}),
        H.ptr, Cdouble(beta), Cint(n_random), Cint(n_steps), res_c
    )
    _check_status(status, "FTLM solver failed")
    return FTLMResult{Float64}(
        res_c[].beta,
        res_c[].partition_function,
        res_c[].internal_energy,
        res_c[].specific_heat
    )
end

function ftlm(
    H::MatrixFreeHamiltonian{Float32};
    beta::Real=1.0,
    n_random::Integer=10,
    n_steps::Integer=50
)::FTLMResult{Float32}
    res_c = Ref{FTLMResultFP32C}(FTLMResultFP32C(0.0f0, 0.0f0, 0.0f0, 0.0f0))
    status = ccall(
        (:qkrylov_ftlm_fp32, libqkrylov),
        Cint,
        (Ptr{Cvoid}, Cfloat, Cint, Cint, Ref{FTLMResultFP32C}),
        H.ptr, Cfloat(beta), Cint(n_random), Cint(n_steps), res_c
    )
    _check_status(status, "FTLM solver failed")
    return FTLMResult{Float32}(
        res_c[].beta,
        res_c[].partition_function,
        res_c[].internal_energy,
        res_c[].specific_heat
    )
end

struct FTLMSweepResult{T<:Real}
    beta_grid::Vector{T}
    partition_functions::Vector{T}
    free_energies::Vector{T}
    internal_energies::Vector{T}
    specific_heats::Vector{T}
    entropies::Vector{T}
    observable_expectations::Vector{Vector{T}}
    observable_errors::Vector{Vector{T}}
end

function Base.getproperty(res::FTLMSweepResult, s::Symbol)
    if s === :partition_function
        return getfield(res, :partition_functions)[1]
    elseif s === :internal_energy
        return getfield(res, :internal_energies)[1]
    elseif s === :specific_heat
        return getfield(res, :specific_heats)[1]
    elseif s === :free_energy
        return getfield(res, :free_energies)[1]
    elseif s === :entropy
        return getfield(res, :entropies)[1]
    elseif s === :beta
        return getfield(res, :beta_grid)[1]
    else
        return getfield(res, s)
    end
end

function ftlm_sweep(
    H::MatrixFreeHamiltonian{Float64};
    betas::AbstractVector{<:Real}=[1.0],
    observables::Vector{<:MatrixFreeHamiltonian{Float64}}=MatrixFreeHamiltonian{Float64}[],
    n_random::Integer=50,
    n_steps::Integer=100,
    seed::Integer=42
)::FTLMSweepResult{Float64}
    nb = length(betas)
    n_obs = length(observables)
    beta_arr = Vector{Float64}(betas)
    obs_ptrs = [obs.ptr for obs in observables]

    res_c = Ref{FTLMSweepResultFP64C}()
    status = ccall(
        (:qkrylov_ftlm_sweep_fp64, libqkrylov),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}, Cint, Ptr{Ptr{Cvoid}}, Cint, Cint, Cint, Culonglong, Ref{FTLMSweepResultFP64C}),
        H.ptr, pointer(beta_arr), Cint(nb), isempty(obs_ptrs) ? C_NULL : pointer(obs_ptrs), Cint(n_obs), Cint(n_random), Cint(n_steps), Culonglong(seed), res_c
    )
    _check_status(status, "FTLM sweep failed")

    raw = res_c[]
    b_grid = copy(unsafe_wrap(Array, raw.beta_grid, nb))
    z_arr  = copy(unsafe_wrap(Array, raw.partition_functions, nb))
    f_arr  = copy(unsafe_wrap(Array, raw.free_energies, nb))
    e_arr  = copy(unsafe_wrap(Array, raw.internal_energies, nb))
    cv_arr = copy(unsafe_wrap(Array, raw.specific_heats, nb))
    s_arr  = copy(unsafe_wrap(Array, raw.entropies, nb))

    obs_exp = Vector{Vector{Float64}}(undef, n_obs)
    obs_err = Vector{Vector{Float64}}(undef, n_obs)
    if n_obs > 0
        raw_obs = unsafe_wrap(Array, raw.observable_expectations, (nb, n_obs))
        raw_err = unsafe_wrap(Array, raw.observable_errors, (nb, n_obs))
        for oi in 1:n_obs
            obs_exp[oi] = copy(raw_obs[:, oi])
            obs_err[oi] = copy(raw_err[:, oi])
        end
    end

    ccall((:qkrylov_ftlm_sweep_result_free_fp64, libqkrylov), Cvoid, (Ref{FTLMSweepResultFP64C},), res_c)

    return FTLMSweepResult{Float64}(b_grid, z_arr, f_arr, e_arr, cv_arr, s_arr, obs_exp, obs_err)
end

function ftlm_sweep(
    H::MatrixFreeHamiltonian{Float32};
    betas::AbstractVector{<:Real}=[1.0],
    observables::Vector{<:MatrixFreeHamiltonian{Float32}}=MatrixFreeHamiltonian{Float32}[],
    n_random::Integer=50,
    n_steps::Integer=100,
    seed::Integer=42
)::FTLMSweepResult{Float32}
    nb = length(betas)
    n_obs = length(observables)
    beta_arr = Vector{Float32}(betas)
    obs_ptrs = [obs.ptr for obs in observables]

    res_c = Ref{FTLMSweepResultFP32C}()
    status = ccall(
        (:qkrylov_ftlm_sweep_fp32, libqkrylov),
        Cint,
        (Ptr{Cvoid}, Ptr{Cfloat}, Cint, Ptr{Ptr{Cvoid}}, Cint, Cint, Cint, Culonglong, Ref{FTLMSweepResultFP32C}),
        H.ptr, pointer(beta_arr), Cint(nb), isempty(obs_ptrs) ? C_NULL : pointer(obs_ptrs), Cint(n_obs), Cint(n_random), Cint(n_steps), Culonglong(seed), res_c
    )
    _check_status(status, "FTLM sweep failed")

    raw = res_c[]
    b_grid = copy(unsafe_wrap(Array, raw.beta_grid, nb))
    z_arr  = copy(unsafe_wrap(Array, raw.partition_functions, nb))
    f_arr  = copy(unsafe_wrap(Array, raw.free_energies, nb))
    e_arr  = copy(unsafe_wrap(Array, raw.internal_energies, nb))
    cv_arr = copy(unsafe_wrap(Array, raw.specific_heats, nb))
    s_arr  = copy(unsafe_wrap(Array, raw.entropies, nb))

    obs_exp = Vector{Vector{Float32}}(undef, n_obs)
    obs_err = Vector{Vector{Float32}}(undef, n_obs)
    if n_obs > 0
        raw_obs = unsafe_wrap(Array, raw.observable_expectations, (nb, n_obs))
        raw_err = unsafe_wrap(Array, raw.observable_errors, (nb, n_obs))
        for oi in 1:n_obs
            obs_exp[oi] = copy(raw_obs[:, oi])
            obs_err[oi] = copy(raw_err[:, oi])
        end
    end

    ccall((:qkrylov_ftlm_sweep_result_free_fp32, libqkrylov), Cvoid, (Ref{FTLMSweepResultFP32C},), res_c)

    return FTLMSweepResult{Float32}(b_grid, z_arr, f_arr, e_arr, cv_arr, s_arr, obs_exp, obs_err)
end

# Correction Vector Spectroscopy Solver
struct CorrectionVectorResult{T<:Real}
    spectral_function::T
    iterations::Int
    converged::Bool
    _vector::Union{Vector{Complex{T}}, Nothing}
    has_vector::Bool

    function CorrectionVectorResult(spec::T, iterations::Integer, converged::Bool, vec::Union{Vector{Complex{T}}, Nothing}=nothing) where {T<:Real}
        return new{T}(spec, Int(iterations), converged, vec, vec !== nothing)
    end
end

function Base.getproperty(res::CorrectionVectorResult, sym::Symbol)
    if sym === :vector || sym === :correction_vector
        if !getfield(res, :has_vector) || getfield(res, :_vector) === nothing
            error("Correction vector was not computed. Pass `return_vector=true` to `solver_correction_vector`.")
        end
        return getfield(res, :_vector)
    end
    return getfield(res, sym)
end

function Base.propertynames(res::CorrectionVectorResult, private::Bool=false)
    return private ? fieldnames(CorrectionVectorResult) : (:spectral_function, :iterations, :converged, :vector, :correction_vector)
end

function solver_correction_vector(
    H::MatrixFreeHamiltonian{Float64},
    op_psi0::AbstractVector{<:Number};
    e0::Real,
    omega::Real,
    eta::Real,
    maxiter::Integer=100,
    tol::Real=1e-8,
    return_vector::Bool=false
)::CorrectionVectorResult{Float64}
    dim = Int(dimension(H))
    @assert length(op_psi0) == dim "Input vector size $(length(op_psi0)) does not match Hamiltonian dimension $dim"

    op_psi0_c = (op_psi0 isa Vector{ComplexF64}) ? op_psi0 : Vector{ComplexF64}(op_psi0)
    vec_out = return_vector ? Vector{ComplexF64}(undef, dim) : ComplexF64[]
    res_c = Ref{CorrectionVectorResultFP64C}(CorrectionVectorResultFP64C(0.0, 0, 0))

    GC.@preserve op_psi0_c vec_out begin
        vec_ptr = return_vector ? pointer(vec_out) : Ptr{ComplexF64}(C_NULL)
        status = ccall(
            (:qkrylov_solver_correction_vector_fp64, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Ptr{Cdouble}, Cdouble, Cdouble, Cdouble, Cint, Cdouble, Ref{CorrectionVectorResultFP64C}, Ptr{Cdouble}),
            H.ptr, pointer(op_psi0_c), Cdouble(e0), Cdouble(omega), Cdouble(eta), Cint(maxiter), Cdouble(tol), res_c, Ptr{Cdouble}(vec_ptr)
        )
        _check_status(status, "Correction vector solver failed")
    end
    return CorrectionVectorResult(res_c[].spectral_function, Int(res_c[].iterations), res_c[].converged != 0, return_vector ? vec_out : nothing)
end

function solver_correction_vector(
    H::MatrixFreeHamiltonian{Float32},
    op_psi0::AbstractVector{<:Number};
    e0::Real,
    omega::Real,
    eta::Real,
    maxiter::Integer=100,
    tol::Real=1e-5,
    return_vector::Bool=false
)::CorrectionVectorResult{Float32}
    dim = Int(dimension(H))
    @assert length(op_psi0) == dim "Input vector size $(length(op_psi0)) does not match Hamiltonian dimension $dim"

    op_psi0_c = (op_psi0 isa Vector{ComplexF32}) ? op_psi0 : Vector{ComplexF32}(op_psi0)
    vec_out = return_vector ? Vector{ComplexF32}(undef, dim) : ComplexF32[]
    res_c = Ref{CorrectionVectorResultFP32C}(CorrectionVectorResultFP32C(0.0f0, 0, 0))

    GC.@preserve op_psi0_c vec_out begin
        vec_ptr = return_vector ? pointer(vec_out) : Ptr{ComplexF32}(C_NULL)
        status = ccall(
            (:qkrylov_solver_correction_vector_fp32, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Ptr{Cfloat}, Cfloat, Cfloat, Cfloat, Cint, Cfloat, Ref{CorrectionVectorResultFP32C}, Ptr{Cfloat}),
            H.ptr, pointer(op_psi0_c), Cfloat(e0), Cfloat(omega), Cfloat(eta), Cint(maxiter), Cfloat(tol), res_c, Ptr{Cfloat}(vec_ptr)
        )
        _check_status(status, "Correction vector solver failed")
    end
    return CorrectionVectorResult(res_c[].spectral_function, Int(res_c[].iterations), res_c[].converged != 0, return_vector ? vec_out : nothing)
end

# -----------------------------------------------------------------------------
# Base.show Formatting for Solver Results
# -----------------------------------------------------------------------------

function Base.show(io::IO, res::LanczosResult{T}) where {T}
    status_str = res.converged ? "converged = true" : "WARNING: maxiter hit without converging!"
    state_str = res.has_state ? ", state = Vector{Complex{$T}}(dim=$(length(res._state)))" : ""
    print(io, "LanczosResult{$T}(energy = $(res.energy), iterations = $(res.iterations), $status_str$state_str)")
end

function Base.show(io::IO, res::ExcitedStatesSolution{T}) where {T}
    n = length(res.eigenvalues)
    has_v = res.eigenvectors !== nothing
    status_str = res.converged ? "converged = true" : "WARNING: maxiter hit without converging!"
    print(io, "ExcitedStatesSolution{$T}(n_eig = $n, energies = $(res.eigenvalues), iterations = $(res.iterations), $status_str, has_eigenvectors = $has_v)")
end

function Base.show(io::IO, res::ContinuedFractionResult{T}) where {T}
    n = length(res.alphas)
    print(io, "ContinuedFractionResult{$T}(n_coeffs = $n, norm_phi0 = $(res.norm_phi0))")
end

function Base.show(io::IO, res::FTLMResult{T}) where {T}
    print(io, "FTLMResult{$T}(beta = $(res.beta), Z = $(res.partition_function), E = $(res.internal_energy), Cv = $(res.specific_heat))")
end

function Base.show(io::IO, res::CorrectionVectorResult{T}) where {T}
    status_str = res.converged ? "converged = true" : "WARNING: maxiter hit without converging!"
    vec_str = res.has_vector ? ", vector = Vector{Complex{$T}}(dim=$(length(res._vector)))" : ""
    print(io, "CorrectionVectorResult{$T}(S = $(res.spectral_function), iterations = $(res.iterations), $status_str$vec_str)")
end
