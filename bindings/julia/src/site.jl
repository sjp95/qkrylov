# Site wrapper structs

abstract type AbstractSite end

mutable struct SpinHalfSite <: AbstractSite
    ptr::Ptr{Cvoid}

    function SpinHalfSite()
        ptr = ccall((:qkrylov_spinhalf_site_create, libqkrylov), Ptr{Cvoid}, ())
        ptr == C_NULL && error("Failed to create SpinHalfSite")
        obj = new(ptr)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall((:qkrylov_site_destroy, libqkrylov), Cvoid, (Ptr{Cvoid},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

mutable struct FermionSite <: AbstractSite
    ptr::Ptr{Cvoid}

    function FermionSite()
        ptr = ccall((:qkrylov_fermion_site_create, libqkrylov), Ptr{Cvoid}, ())
        ptr == C_NULL && error("Failed to create FermionSite")
        obj = new(ptr)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall((:qkrylov_site_destroy, libqkrylov), Cvoid, (Ptr{Cvoid},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

mutable struct HubbardSite <: AbstractSite
    ptr::Ptr{Cvoid}

    function HubbardSite()
        ptr = ccall((:qkrylov_hubbard_site_create, libqkrylov), Ptr{Cvoid}, ())
        ptr == C_NULL && error("Failed to create HubbardSite")
        obj = new(ptr)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall((:qkrylov_site_destroy, libqkrylov), Cvoid, (Ptr{Cvoid},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

mutable struct TJSite <: AbstractSite
    ptr::Ptr{Cvoid}

    function TJSite()
        ptr = ccall((:qkrylov_tj_site_create, libqkrylov), Ptr{Cvoid}, ())
        ptr == C_NULL && error("Failed to create TJSite")
        obj = new(ptr)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall((:qkrylov_site_destroy, libqkrylov), Cvoid, (Ptr{Cvoid},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

mutable struct SpinSSite <: AbstractSite
    ptr::Ptr{Cvoid}
    spin_s::Float64

    function SpinSSite(S::Real)
        ptr = ccall((:qkrylov_site_create_spin_s, libqkrylov), Ptr{Cvoid}, (Cdouble,), Cdouble(S))
        ptr == C_NULL && error("Failed to create SpinSSite with S=$S")
        obj = new(ptr, Float64(S))
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall((:qkrylov_site_destroy, libqkrylov), Cvoid, (Ptr{Cvoid},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

Base.show(io::IO, ::SpinHalfSite) = print(io, "SpinHalfSite(dim = 2, states = [↑, ↓])")
Base.show(io::IO, s::SpinSSite)    = print(io, "SpinSSite(S = $(s.spin_s), dim = $(round(Int, 2*s.spin_s + 1)))")
Base.show(io::IO, ::FermionSite)  = print(io, "FermionSite(dim = 2, states = [0, 1])")
Base.show(io::IO, ::HubbardSite)  = print(io, "HubbardSite(dim = 4, states = [0, ↑, ↓, ↑↓])")
Base.show(io::IO, ::TJSite)       = print(io, "TJSite(dim = 3, states = [0, ↑, ↓])")

struct LocalAction
    valid::Bool
    new_state::UInt64
    matrix_element::ComplexF64
end

function apply(site::AbstractSite, op::AbstractString, site_idx::Integer, state::Unsigned)::LocalAction
    action_c = Ref{LocalActionC}()
    status = ccall(
        (:qkrylov_site_apply, libqkrylov),
        Cint,
        (Ptr{Cvoid}, Cstring, Cint, UInt64, Ref{LocalActionC}),
        site.ptr, string(op), Cint(site_idx), UInt64(state), action_c
    )
    _check_status(status, "Failed to apply operator '$op' on site $site_idx")
    act = action_c[]
    return LocalAction(act.valid != 0, act.new_state, ComplexF64(act.matrix_element_re, act.matrix_element_im))
end

function spin(s::AbstractSite)::Float64
    spin_out = Ref{Cdouble}(0.0)
    status = ccall((:qkrylov_site_get_spin, libqkrylov), Cint, (Ptr{Cvoid}, Ref{Cdouble}), s.ptr, spin_out)
    _check_status(status, "Failed to query site spin")
    return spin_out[]
end

function dimension_per_site(s::AbstractSite)::Int
    d_out = Ref{Cint}(0)
    status = ccall((:qkrylov_site_get_dimension_per_site, libqkrylov), Cint, (Ptr{Cvoid}, Ref{Cint}), s.ptr, d_out)
    _check_status(status, "Failed to query site dimension per site")
    return Int(d_out[])
end

function site_type(s::AbstractSite)::Symbol
    t_out = Ref{Cint}(0)
    status = ccall((:qkrylov_site_get_type, libqkrylov), Cint, (Ptr{Cvoid}, Ref{Cint}), s.ptr, t_out)
    _check_status(status, "Failed to query site type")
    t = t_out[]
    t == 0 && return :SpinHalf
    t == 1 && return :SpinS
    t == 2 && return :Fermion
    t == 3 && return :Hubbard
    t == 4 && return :TJ
    return :Unknown
end


