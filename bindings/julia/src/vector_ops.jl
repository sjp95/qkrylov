# Vector operations using Kokkos parallel kernels

"""
    vector_dot(x::AbstractVector{Complex{T}}, y::AbstractVector{Complex{T}}) where {T<:Union{Float64, Float32}} -> Complex{T}

Computes the inner product ⟨x|y⟩ = sum_i conj(x_i) * y_i using accelerated Kokkos parallel reductions.
"""
function vector_dot(x::AbstractVector{Complex{T}}, y::AbstractVector{Complex{T}})::Complex{T} where {T<:Union{Float64, Float32}}
    length(x) == length(y) || throw(DimensionMismatch("Vectors must have the same length (got $(length(x)) and $(length(y)))"))
    dim = UInt64(length(x))
    if dim == 0
        return zero(Complex{T})
    end
    xc = x isa Vector{Complex{T}} ? x : Vector{Complex{T}}(x)
    yc = y isa Vector{Complex{T}} ? y : Vector{Complex{T}}(y)
    re = Ref{T}(zero(T))
    im = Ref{T}(zero(T))
    GC.@preserve xc yc begin
        if T === Float64
            status = ccall((:qkrylov_vector_dot_fp64, libqkrylov), Cint,
                (UInt64, Ptr{Cdouble}, Ptr{Cdouble}, Ref{Cdouble}, Ref{Cdouble}),
                dim, pointer(xc), pointer(yc), re, im)
        else
            status = ccall((:qkrylov_vector_dot_fp32, libqkrylov), Cint,
                (UInt64, Ptr{Cfloat}, Ptr{Cfloat}, Ref{Cfloat}, Ref{Cfloat}),
                dim, pointer(xc), pointer(yc), re, im)
        end
        _check_status(status, "qkrylov_vector_dot failed")
    end
    return Complex{T}(re[], im[])
end

"""
    vector_norm(x::AbstractVector{Complex{T}}) where {T<:Union{Float64, Float32}} -> T

Computes the Euclidean 2-norm ||x|| using accelerated Kokkos parallel reductions.
"""
function vector_norm(x::AbstractVector{Complex{T}})::T where {T<:Union{Float64, Float32}}
    dim = UInt64(length(x))
    if dim == 0
        return zero(T)
    end
    xc = x isa Vector{Complex{T}} ? x : Vector{Complex{T}}(x)
    res = Ref{T}(zero(T))
    GC.@preserve xc begin
        if T === Float64
            status = ccall((:qkrylov_vector_norm_fp64, libqkrylov), Cint,
                (UInt64, Ptr{Cdouble}, Ref{Cdouble}),
                dim, pointer(xc), res)
        else
            status = ccall((:qkrylov_vector_norm_fp32, libqkrylov), Cint,
                (UInt64, Ptr{Cfloat}, Ref{Cfloat}),
                dim, pointer(xc), res)
        end
        _check_status(status, "qkrylov_vector_norm failed")
    end
    return res[]
end

"""
    vector_axpy!(a::Number, x::AbstractVector{Complex{T}}, y::Vector{Complex{T}}) where {T<:Union{Float64, Float32}} -> Vector{Complex{T}}

Computes y = a * x + y in-place using Kokkos parallel dispatch.
"""
function vector_axpy!(a::Number, x::AbstractVector{Complex{T}}, y::Vector{Complex{T}})::Vector{Complex{T}} where {T<:Union{Float64, Float32}}
    length(x) == length(y) || throw(DimensionMismatch("Vectors must have the same length (got $(length(x)) and $(length(y)))"))
    dim = UInt64(length(x))
    if dim == 0
        return y
    end
    xc = x isa Vector{Complex{T}} ? x : Vector{Complex{T}}(x)
    a_c = Complex{T}(a)
    GC.@preserve xc y begin
        if T === Float64
            status = ccall((:qkrylov_vector_axpy_fp64, libqkrylov), Cint,
                (UInt64, Cdouble, Cdouble, Ptr{Cdouble}, Ptr{Cdouble}),
                dim, real(a_c), imag(a_c), pointer(xc), pointer(y))
        else
            status = ccall((:qkrylov_vector_axpy_fp32, libqkrylov), Cint,
                (UInt64, Cfloat, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}),
                dim, real(a_c), imag(a_c), pointer(xc), pointer(y))
        end
        _check_status(status, "qkrylov_vector_axpy failed")
    end
    return y
end

"""
    vector_scal!(a::Number, x::Vector{Complex{T}}) where {T<:Union{Float64, Float32}} -> Vector{Complex{T}}

Computes x = a * x in-place using Kokkos parallel dispatch.
"""
function vector_scal!(a::Number, x::Vector{Complex{T}})::Vector{Complex{T}} where {T<:Union{Float64, Float32}}
    dim = UInt64(length(x))
    if dim == 0
        return x
    end
    a_c = Complex{T}(a)
    GC.@preserve x begin
        if T === Float64
            status = ccall((:qkrylov_vector_scal_fp64, libqkrylov), Cint,
                (UInt64, Cdouble, Cdouble, Ptr{Cdouble}),
                dim, real(a_c), imag(a_c), pointer(x))
        else
            status = ccall((:qkrylov_vector_scal_fp32, libqkrylov), Cint,
                (UInt64, Cfloat, Cfloat, Ptr{Cfloat}),
                dim, real(a_c), imag(a_c), pointer(x))
        end
        _check_status(status, "qkrylov_vector_scal failed")
    end
    return x
end

"""
    vector_normalize!(x::Vector{Complex{T}}) where {T<:Union{Float64, Float32}} -> Vector{Complex{T}}

Normalizes vector x in-place to unit length ||x|| = 1 using Kokkos parallel kernels.
"""
function vector_normalize!(x::Vector{Complex{T}})::Vector{Complex{T}} where {T<:Union{Float64, Float32}}
    dim = UInt64(length(x))
    if dim == 0
        return x
    end
    GC.@preserve x begin
        if T === Float64
            status = ccall((:qkrylov_vector_normalize_fp64, libqkrylov), Cint,
                (UInt64, Ptr{Cdouble}),
                dim, pointer(x))
        else
            status = ccall((:qkrylov_vector_normalize_fp32, libqkrylov), Cint,
                (UInt64, Ptr{Cfloat}),
                dim, pointer(x))
        end
        _check_status(status, "qkrylov_vector_normalize failed")
    end
    return x
end

"""
    vector_zero_fill!(x::Vector{Complex{T}}) where {T<:Union{Float64, Float32}} -> Vector{Complex{T}}

Sets all elements of vector x to zero in-place using Kokkos parallel kernels.
"""
function vector_zero_fill!(x::Vector{Complex{T}})::Vector{Complex{T}} where {T<:Union{Float64, Float32}}
    dim = UInt64(length(x))
    if dim == 0
        return x
    end
    GC.@preserve x begin
        if T === Float64
            status = ccall((:qkrylov_vector_zero_fill_fp64, libqkrylov), Cint,
                (UInt64, Ptr{Cdouble}),
                dim, pointer(x))
        else
            status = ccall((:qkrylov_vector_zero_fill_fp32, libqkrylov), Cint,
                (UInt64, Ptr{Cfloat}),
                dim, pointer(x))
        end
        _check_status(status, "qkrylov_vector_zero_fill failed")
    end
    return x
end

"""
    vector_copy!(dst::Vector{Complex{T}}, src::AbstractVector{Complex{T}}) where {T<:Union{Float64, Float32}} -> Vector{Complex{T}}

Copies elements from src to dst using Kokkos parallel deep copy.
"""
function vector_copy!(dst::Vector{Complex{T}}, src::AbstractVector{Complex{T}})::Vector{Complex{T}} where {T<:Union{Float64, Float32}}
    length(src) == length(dst) || throw(DimensionMismatch("Vectors must have the same length (got $(length(src)) and $(length(dst)))"))
    dim = UInt64(length(src))
    if dim == 0
        return dst
    end
    s_vec = src isa Vector{Complex{T}} ? src : Vector{Complex{T}}(src)
    GC.@preserve dst s_vec begin
        if T === Float64
            status = ccall((:qkrylov_vector_copy_fp64, libqkrylov), Cint,
                (UInt64, Ptr{Cdouble}, Ptr{Cdouble}),
                dim, pointer(s_vec), pointer(dst))
        else
            status = ccall((:qkrylov_vector_copy_fp32, libqkrylov), Cint,
                (UInt64, Ptr{Cfloat}, Ptr{Cfloat}),
                dim, pointer(s_vec), pointer(dst))
        end
        _check_status(status, "qkrylov_vector_copy failed")
    end
    return dst
end

# ==============================================================================
# DeviceVector BLAS-1 Operations (Kokkos parallel kernels on device memory)
# ==============================================================================

function vector_dot(x::DeviceVector{T}, y::DeviceVector{T})::Complex{T} where {T<:Union{Float64, Float32}}
    length(x) == length(y) || throw(DimensionMismatch("DeviceVectors must have the same length (got $(length(x)) and $(length(y)))"))
    re = Ref{T}(zero(T))
    im = Ref{T}(zero(T))
    status = if T === Float64
        ccall((:qkrylov_device_vector_dot_fp64, libqkrylov), Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{Cdouble}, Ref{Cdouble}),
            x.ptr, y.ptr, re, im)
    else
        ccall((:qkrylov_device_vector_dot_fp32, libqkrylov), Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{Cfloat}, Ref{Cfloat}),
            x.ptr, y.ptr, re, im)
    end
    _check_status(status, "qkrylov_device_vector_dot failed")
    return Complex{T}(re[], im[])
end

function vector_norm(x::DeviceVector{T})::T where {T<:Union{Float64, Float32}}
    res = Ref{T}(zero(T))
    status = if T === Float64
        ccall((:qkrylov_device_vector_norm_fp64, libqkrylov), Cint,
            (Ptr{Cvoid}, Ref{Cdouble}),
            x.ptr, res)
    else
        ccall((:qkrylov_device_vector_norm_fp32, libqkrylov), Cint,
            (Ptr{Cvoid}, Ref{Cfloat}),
            x.ptr, res)
    end
    _check_status(status, "qkrylov_device_vector_norm failed")
    return res[]
end

function vector_axpy!(a::Number, x::DeviceVector{T}, y::DeviceVector{T})::DeviceVector{T} where {T<:Union{Float64, Float32}}
    length(x) == length(y) || throw(DimensionMismatch("DeviceVectors must have the same length (got $(length(x)) and $(length(y)))"))
    a_c = Complex{T}(a)
    status = if T === Float64
        ccall((:qkrylov_device_vector_axpy_fp64, libqkrylov), Cint,
            (Cdouble, Cdouble, Ptr{Cvoid}, Ptr{Cvoid}),
            real(a_c), imag(a_c), x.ptr, y.ptr)
    else
        ccall((:qkrylov_device_vector_axpy_fp32, libqkrylov), Cint,
            (Cfloat, Cfloat, Ptr{Cvoid}, Ptr{Cvoid}),
            real(a_c), imag(a_c), x.ptr, y.ptr)
    end
    _check_status(status, "qkrylov_device_vector_axpy failed")
    return y
end

function vector_scal!(a::Number, x::DeviceVector{T})::DeviceVector{T} where {T<:Union{Float64, Float32}}
    a_c = Complex{T}(a)
    status = if T === Float64
        ccall((:qkrylov_device_vector_scal_fp64, libqkrylov), Cint,
            (Cdouble, Cdouble, Ptr{Cvoid}),
            real(a_c), imag(a_c), x.ptr)
    else
        ccall((:qkrylov_device_vector_scal_fp32, libqkrylov), Cint,
            (Cfloat, Cfloat, Ptr{Cvoid}),
            real(a_c), imag(a_c), x.ptr)
    end
    _check_status(status, "qkrylov_device_vector_scal failed")
    return x
end

function vector_normalize!(x::DeviceVector{T})::DeviceVector{T} where {T<:Union{Float64, Float32}}
    status = if T === Float64
        ccall((:qkrylov_device_vector_normalize_fp64, libqkrylov), Cint,
            (Ptr{Cvoid},), x.ptr)
    else
        ccall((:qkrylov_device_vector_normalize_fp32, libqkrylov), Cint,
            (Ptr{Cvoid},), x.ptr)
    end
    _check_status(status, "qkrylov_device_vector_normalize failed")
    return x
end

function vector_zero_fill!(x::DeviceVector{T})::DeviceVector{T} where {T<:Union{Float64, Float32}}
    status = if T === Float64
        ccall((:qkrylov_device_vector_zero_fill_fp64, libqkrylov), Cint,
            (Ptr{Cvoid},), x.ptr)
    else
        ccall((:qkrylov_device_vector_zero_fill_fp32, libqkrylov), Cint,
            (Ptr{Cvoid},), x.ptr)
    end
    _check_status(status, "qkrylov_device_vector_zero_fill failed")
    return x
end

function vector_copy!(dst::DeviceVector{T}, src::DeviceVector{T})::DeviceVector{T} where {T<:Union{Float64, Float32}}
    length(dst) == length(src) || throw(DimensionMismatch("DeviceVectors must have the same length (got $(length(dst)) and $(length(src)))"))
    status = if T === Float64
        ccall((:qkrylov_device_vector_copy_fp64, libqkrylov), Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}), src.ptr, dst.ptr)
    else
        ccall((:qkrylov_device_vector_copy_fp32, libqkrylov), Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}), src.ptr, dst.ptr)
    end
    _check_status(status, "qkrylov_device_vector_copy failed")
    return dst
end
