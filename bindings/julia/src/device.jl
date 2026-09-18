# Device management and hardware query wrappers
using Libdl

# Hardware Device Traits
abstract type AbstractDevice end
struct CPUDevice   <: AbstractDevice end
struct CUDADevice  <: AbstractDevice end
struct HIPDevice   <: AbstractDevice end
struct SYCLDevice  <: AbstractDevice end

device_string(::CPUDevice)  = "cpu"
device_string(::CUDADevice) = "cuda"
device_string(::HIPDevice)  = "hip"
device_string(::SYCLDevice) = "sycl"
device_string(s::AbstractString) = String(s)

Base.string(d::AbstractDevice) = device_string(d)
Base.:(==)(::CPUDevice, s::AbstractString) = lowercase(s) == "cpu"
Base.:(==)(s::AbstractString, d::AbstractDevice) = (d == s)
Base.:(==)(::CUDADevice, s::AbstractString) = (lowercase(s) == "cuda" || lowercase(s) == "gpu")
Base.:(==)(::HIPDevice, s::AbstractString) = lowercase(s) == "hip"
Base.:(==)(::SYCLDevice, s::AbstractString) = lowercase(s) == "sycl"

function _has_symbol(sym::Symbol)::Bool
    try
        h = Libdl.dlopen(libqkrylov, Libdl.RTLD_LAZY | Libdl.RTLD_LOCAL)
        return Libdl.dlsym_e(h, sym) != C_NULL
    catch
        return false
    end
end

"""
    is_gpu_build() -> Bool

Return `true` if the underlying `libqkrylov` binary was compiled with GPU acceleration
(CUDA, HIP, or SYCL), or `false` for a CPU-only build.
"""
function is_gpu_build()::Bool
    if !_has_symbol(:qkrylov_is_gpu_build)
        return false
    end
    return ccall((:qkrylov_is_gpu_build, libqkrylov), Cint, ()) != 0
end

"""
    find_gpu() -> Union{String, Nothing}

Return the name of the compiled GPU backend ("cuda", "hip", "sycl") if available,
or `nothing` if built for CPU only. (Matches Python API `qkrylov.find_gpu()`).
"""
function find_gpu()::Union{String, Nothing}
    if !_has_symbol(:qkrylov_find_gpu)
        return nothing
    end
    ptr = ccall((:qkrylov_find_gpu, libqkrylov), Cstring, ())
    return ptr == C_NULL ? nothing : unsafe_string(ptr)
end

"""
    gpu_count() -> Int

Return the number of available physical GPUs detected on the system.
(Matches Python API `qkrylov.gpu_count()`).
"""
function gpu_count()::Int
    if !_has_symbol(:qkrylov_gpu_count)
        return 0
    end
    return Int(ccall((:qkrylov_gpu_count, libqkrylov), Cint, ()))
end

"""
    initialize_device!(device::AbstractString="cpu")

Explicitly initialize Kokkos execution spaces for a targeted device (e.g. "cpu", "cuda:0").
"""
function initialize_device!(device::AbstractString="cpu")
    if !_has_symbol(:qkrylov_initialize_device)
        return nothing
    end
    status = ccall((:qkrylov_initialize_device, libqkrylov), Cint, (Cstring,), device)
    status != QKRYLOV_SUCCESS && error("Failed to initialize target device: $device with status code $status")
    return nothing
end

"""
    DeviceVector{T}(dim::Integer) where {T<:Union{Float32, Float64}}
    DeviceVector(dim::Integer; precision::Type{T}=Float64)
    DeviceVector(src::AbstractVector)

Opaque handle to hardware-accelerated device-resident memory (GPU VRAM or host Kokkos View).
Enables zero-copy matrix-vector multiplication (`H * v_dev` or `mul!(y_dev, H, x_dev)`) and pure
device BLAS-1 operations without CPU host staging overhead.
"""
mutable struct DeviceVector{T<:Union{Float32, Float64}}
    ptr::Ptr{Cvoid}
    dim::Int
    precision::Type{T}

    function DeviceVector{T}(dim::Integer) where {T<:Union{Float32, Float64}}
        dim < 0 && throw(ArgumentError("DeviceVector dimension must be non-negative, got $dim"))
        ptr = if T === Float32
            ccall((:qkrylov_device_vector_create_fp32, libqkrylov), Ptr{Cvoid}, (UInt64,), UInt64(dim))
        else
            ccall((:qkrylov_device_vector_create_fp64, libqkrylov), Ptr{Cvoid}, (UInt64,), UInt64(dim))
        end
        if ptr == C_NULL
            err = get_last_error_message()
            error("Failed to allocate DeviceVector{$T} of dimension $dim: $err")
        end
        obj = new{T}(ptr, Int(dim), T)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall((:qkrylov_device_vector_destroy, libqkrylov), Cvoid, (Ptr{Cvoid},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

DeviceVector(dim::Integer; precision::Type{T}=Float64) where {T<:Union{Float32, Float64}} = DeviceVector{precision}(dim)

Base.length(v::DeviceVector) = v.dim
Base.size(v::DeviceVector) = (v.dim,)
Base.size(v::DeviceVector, d::Integer) = d == 1 ? v.dim : 1
Base.eltype(::Type{DeviceVector{T}}) where {T} = Complex{T}
Base.eltype(::DeviceVector{T}) where {T} = Complex{T}
Base.pointer(v::DeviceVector{T}) where {T} = Ptr{Complex{T}}(ccall((:qkrylov_device_vector_data, libqkrylov), Ptr{Cvoid}, (Ptr{Cvoid},), v.ptr))

function Base.show(io::IO, v::DeviceVector{T}) where {T}
    print(io, "DeviceVector{$T}(dim = $(v.dim))")
end

# Constructor from host vector
function DeviceVector(src::AbstractVector{Complex{T}}) where {T<:Union{Float32, Float64}}
    dev_vec = DeviceVector{T}(length(src))
    copyto!(dev_vec, src)
    return dev_vec
end

function DeviceVector{T}(src::AbstractVector{<:Number}) where {T<:Union{Float32, Float64}}
    src_c = (src isa Vector{Complex{T}}) ? src : Vector{Complex{T}}(src)
    dev_vec = DeviceVector{T}(length(src_c))
    copyto!(dev_vec, src_c)
    return dev_vec
end

# Staging copy: Host -> Device
function Base.copyto!(dst::DeviceVector{Float64}, src::AbstractVector{<:Number})
    length(dst) == length(src) || throw(DimensionMismatch("DeviceVector length $(length(dst)) != source length $(length(src))"))
    src_c = (src isa Vector{ComplexF64}) ? src : Vector{ComplexF64}(src)
    GC.@preserve src_c begin
        status = ccall((:qkrylov_device_vector_copy_from_host_fp64, libqkrylov), Cint, (Ptr{Cvoid}, Ptr{Cdouble}), dst.ptr, pointer(src_c))
        _check_status(status, "copyto!(DeviceVector{Float64}, Vector) failed")
    end
    return dst
end

function Base.copyto!(dst::DeviceVector{Float32}, src::AbstractVector{<:Number})
    length(dst) == length(src) || throw(DimensionMismatch("DeviceVector length $(length(dst)) != source length $(length(src))"))
    src_c = (src isa Vector{ComplexF32}) ? src : Vector{ComplexF32}(src)
    GC.@preserve src_c begin
        status = ccall((:qkrylov_device_vector_copy_from_host_fp32, libqkrylov), Cint, (Ptr{Cvoid}, Ptr{Cfloat}), dst.ptr, pointer(src_c))
        _check_status(status, "copyto!(DeviceVector{Float32}, Vector) failed")
    end
    return dst
end

# Staging copy: Device -> Host
function Base.copyto!(dst::Vector{ComplexF64}, src::DeviceVector{Float64})
    length(dst) == length(src) || throw(DimensionMismatch("destination length $(length(dst)) != DeviceVector length $(length(src))"))
    GC.@preserve dst begin
        status = ccall((:qkrylov_device_vector_copy_to_host_fp64, libqkrylov), Cint, (Ptr{Cvoid}, Ptr{Cdouble}), src.ptr, pointer(dst))
        _check_status(status, "copyto!(Vector, DeviceVector{Float64}) failed")
    end
    return dst
end

function Base.copyto!(dst::Vector{ComplexF32}, src::DeviceVector{Float32})
    length(dst) == length(src) || throw(DimensionMismatch("destination length $(length(dst)) != DeviceVector length $(length(src))"))
    GC.@preserve dst begin
        status = ccall((:qkrylov_device_vector_copy_to_host_fp32, libqkrylov), Cint, (Ptr{Cvoid}, Ptr{Cfloat}), src.ptr, pointer(dst))
        _check_status(status, "copyto!(Vector, DeviceVector{Float32}) failed")
    end
    return dst
end

# Copy: Device -> Device
function Base.copyto!(dst::DeviceVector{T}, src::DeviceVector{T}) where {T}
    length(dst) == length(src) || throw(DimensionMismatch("destination DeviceVector length $(length(dst)) != source DeviceVector length $(length(src))"))
    status = if T === Float32
        ccall((:qkrylov_device_vector_copy_fp32, libqkrylov), Cint, (Ptr{Cvoid}, Ptr{Cvoid}), src.ptr, dst.ptr)
    else
        ccall((:qkrylov_device_vector_copy_fp64, libqkrylov), Cint, (Ptr{Cvoid}, Ptr{Cvoid}), src.ptr, dst.ptr)
    end
    _check_status(status, "copyto!(DeviceVector, DeviceVector) failed")
    return dst
end

function Base.copy(src::DeviceVector{T})::DeviceVector{T} where {T}
    dst = DeviceVector{T}(length(src))
    copyto!(dst, src)
    return dst
end

Base.Array(src::DeviceVector{T}) where {T} = Vector(src)
function Base.Vector(src::DeviceVector{T}) where {T}
    host_vec = Vector{Complex{T}}(undef, length(src))
    copyto!(host_vec, src)
    return host_vec
end

function Base.Vector{Complex{T}}(src::DeviceVector{T}) where {T}
    return Vector(src)
end
