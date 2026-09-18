# MatrixFreeHamiltonian & operator overloading

mutable struct MatrixFreeHamiltonian{T<:Union{Float32, Float64}}
    ptr::Ptr{Cvoid}
    basis::AbstractBasis
    site::AbstractSite
    opsum::OpSum
    device::AbstractDevice
    precision::Type{T}

    function MatrixFreeHamiltonian{T}(
        basis::AbstractBasis,
        site::AbstractSite,
        opsum::OpSum;
        device::Union{AbstractDevice, AbstractString} = CPUDevice()
    ) where {T<:Union{Float32, Float64}}
        validate!(opsum, nsites(basis))

        dev_str = device_string(device)
        d_lower = lowercase(dev_str)
        if occursin("cuda", d_lower) || occursin("hip", d_lower) || occursin("sycl", d_lower) || d_lower == "gpu"
            if !is_gpu_build()
                throw(ArgumentError("QKrylov was not built with GPU support. Install/compile a GPU build of libqkrylov with Kokkos CUDA/HIP enabled."))
            end
        end

        dev_trait = if device isa AbstractDevice
            device
        else
            if occursin("cuda", d_lower) || d_lower == "gpu"
                CUDADevice()
            elseif occursin("hip", d_lower)
                HIPDevice()
            elseif occursin("sycl", d_lower)
                SYCLDevice()
            else
                CPUDevice()
            end
        end

        ptr = if T === Float32
            ccall(
                (:qkrylov_hamiltonian_create_device_fp32, libqkrylov),
                Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Cstring),
                basis.ptr, site.ptr, opsum.ptr, dev_str
            )
        else
            ccall(
                (:qkrylov_hamiltonian_create_device_fp64, libqkrylov),
                Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Cstring),
                basis.ptr, site.ptr, opsum.ptr, dev_str
            )
        end
        if ptr == C_NULL
            err = get_last_error_message()
            if !isempty(err)
                error("Failed to create MatrixFreeHamiltonian on device: $dev_str with precision: $T: $err")
            else
                error("Failed to create MatrixFreeHamiltonian on device: $dev_str with precision: $T")
            end
        end

        obj = new{T}(ptr, basis, site, opsum, dev_trait, T)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall((:qkrylov_hamiltonian_destroy, libqkrylov), Cvoid, (Ptr{Cvoid},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

function MatrixFreeHamiltonian(
    basis::AbstractBasis,
    site::AbstractSite,
    opsum::OpSum;
    device::Union{AbstractDevice, AbstractString} = CPUDevice(),
    precision::Type{<:Union{Float32, Float64}} = (device isa CUDADevice || device isa HIPDevice || device isa SYCLDevice || occursin("cuda", lowercase(device_string(device))) || occursin("hip", lowercase(device_string(device))) || occursin("sycl", lowercase(device_string(device))) || lowercase(device_string(device)) == "gpu") ? Float32 : Float64
)
    return MatrixFreeHamiltonian{precision}(basis, site, opsum; device=device)
end

default_site(b::SpinHalfBasis) = SpinHalfSite()
default_site(b::SpinSBasis)    = SpinSSite(b.spin_s)
default_site(b::FermionBasis)  = FermionSite()
default_site(b::HubbardBasis)  = HubbardSite()
default_site(b::TJBasis)       = TJSite()
function default_site(b::AbstractBasis)
    error("Cannot automatically infer Site model for basis type $(typeof(b)). Please provide the site argument explicitly: MatrixFreeHamiltonian(basis, site, opsum)")
end

function MatrixFreeHamiltonian(
    basis::AbstractBasis,
    opsum::OpSum;
    device::Union{AbstractDevice, AbstractString} = CPUDevice(),
    precision::Type{<:Union{Float32, Float64}} = (device isa CUDADevice || device isa HIPDevice || device isa SYCLDevice || occursin("cuda", lowercase(device_string(device))) || occursin("hip", lowercase(device_string(device))) || occursin("sycl", lowercase(device_string(device))) || lowercase(device_string(device)) == "gpu") ? Float32 : Float64
)
    site = default_site(basis)
    return MatrixFreeHamiltonian{precision}(basis, site, opsum; device=device)
end

function MatrixFreeHamiltonian{T}(
    basis::AbstractBasis,
    opsum::OpSum;
    device::Union{AbstractDevice, AbstractString} = CPUDevice()
) where {T<:Union{Float32, Float64}}
    site = default_site(basis)
    return MatrixFreeHamiltonian{T}(basis, site, opsum; device=device)
end

function dimension(H::MatrixFreeHamiltonian)::UInt64
    return ccall((:qkrylov_hamiltonian_dimension, libqkrylov), UInt64, (Ptr{Cvoid},), H.ptr)
end

Base.size(H::MatrixFreeHamiltonian) = (Int(dimension(H)), Int(dimension(H)))
Base.size(H::MatrixFreeHamiltonian, d::Integer) = (d == 1 || d == 2) ? Int(dimension(H)) : 1

function Base.:*(H::MatrixFreeHamiltonian{Float64}, x::AbstractVector{<:Number})::Vector{ComplexF64}
    dim = Int(dimension(H))
    @assert length(x) == dim "Input vector size $(length(x)) does not match Hamiltonian dimension $dim"

    x_c = (x isa Vector{ComplexF64}) ? x : Vector{ComplexF64}(x)
    y_c = Vector{ComplexF64}(undef, dim)

    GC.@preserve x_c y_c begin
        status = ccall(
            (:qkrylov_hamiltonian_apply_complex_fp64, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}),
            H.ptr, pointer(x_c), pointer(y_c)
        )
        _check_status(status, "MatrixFreeHamiltonian apply failed")
    end
    return y_c
end

function Base.:*(H::MatrixFreeHamiltonian{Float32}, x::AbstractVector{<:Number})::Vector{ComplexF32}
    dim = Int(dimension(H))
    @assert length(x) == dim "Input vector size $(length(x)) does not match Hamiltonian dimension $dim"

    x_c = (x isa Vector{ComplexF32}) ? x : Vector{ComplexF32}(x)
    y_c = Vector{ComplexF32}(undef, dim)

    GC.@preserve x_c y_c begin
        status = ccall(
            (:qkrylov_hamiltonian_apply_complex_fp32, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat}),
            H.ptr, pointer(x_c), pointer(y_c)
        )
        _check_status(status, "MatrixFreeHamiltonian apply failed")
    end
    return y_c
end

function diagonal(H::MatrixFreeHamiltonian{Float64})::Vector{Float64}
    dim = Int(dimension(H))
    diag_buf = Vector{Float64}(undef, dim)

    GC.@preserve diag_buf begin
        status = ccall(
            (:qkrylov_hamiltonian_diagonal_fp64, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Ptr{Cdouble}),
            H.ptr, pointer(diag_buf)
        )
        _check_status(status, "Failed to extract Hamiltonian diagonal")
    end
    return diag_buf
end

function diagonal(H::MatrixFreeHamiltonian{Float32})::Vector{Float32}
    dim = Int(dimension(H))
    diag_buf = Vector{Float32}(undef, dim)

    GC.@preserve diag_buf begin
        status = ccall(
            (:qkrylov_hamiltonian_diagonal_fp32, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Ptr{Cfloat}),
            H.ptr, pointer(diag_buf)
        )
        _check_status(status, "Failed to extract Hamiltonian diagonal")
    end
    return diag_buf
end

# Pure Device Matrix-Vector Multiplication: y_dev = H * x_dev (zero copies)
function mul!(y::DeviceVector{Float64}, H::MatrixFreeHamiltonian{Float64}, x::DeviceVector{Float64})
    dim = Int(dimension(H))
    (length(x) == dim && length(y) == dim) || throw(DimensionMismatch("Vector dimensions ($(length(x)), $(length(y))) do not match Hamiltonian dimension $dim"))
    status = ccall((:qkrylov_hamiltonian_apply_device_fp64, libqkrylov), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), H.ptr, x.ptr, y.ptr)
    _check_status(status, "qkrylov_hamiltonian_apply_device_fp64 failed")
    return y
end

function mul!(y::DeviceVector{Float32}, H::MatrixFreeHamiltonian{Float32}, x::DeviceVector{Float32})
    dim = Int(dimension(H))
    (length(x) == dim && length(y) == dim) || throw(DimensionMismatch("Vector dimensions ($(length(x)), $(length(y))) do not match Hamiltonian dimension $dim"))
    status = ccall((:qkrylov_hamiltonian_apply_device_fp32, libqkrylov), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), H.ptr, x.ptr, y.ptr)
    _check_status(status, "qkrylov_hamiltonian_apply_device_fp32 failed")
    return y
end

function Base.:*(H::MatrixFreeHamiltonian{T}, x::DeviceVector{T})::DeviceVector{T} where {T}
    y = DeviceVector{T}(length(x))
    mul!(y, H, x)
    return y
end

"""
    diagonal_device(H::MatrixFreeHamiltonian{T}) -> DeviceVector{T}

Extract the diagonal of the Hamiltonian directly into device-resident memory with zero host copies.
"""
function diagonal_device(H::MatrixFreeHamiltonian{T})::DeviceVector{T} where {T}
    dim = Int(dimension(H))
    diag_dev = DeviceVector{T}(dim)
    status = if T === Float32
        ccall((:qkrylov_hamiltonian_diagonal_device_fp32, libqkrylov), Cint, (Ptr{Cvoid}, Ptr{Cvoid}), H.ptr, diag_dev.ptr)
    else
        ccall((:qkrylov_hamiltonian_diagonal_device_fp64, libqkrylov), Cint, (Ptr{Cvoid}, Ptr{Cvoid}), H.ptr, diag_dev.ptr)
    end
    _check_status(status, "Failed to extract device Hamiltonian diagonal")
    return diag_dev
end

function Base.show(io::IO, H::MatrixFreeHamiltonian{T}) where {T}
    d = dimension(H)
    print(io, "MatrixFreeHamiltonian{$T}(dim = $d, device = \"$(H.device)\", basis = $(H.basis))")
end
