# Sector symmetry wrapper

mutable struct Sector
    ptr::Ptr{Cvoid}

    function Sector()
        ptr = ccall((:qkrylov_sector_create, libqkrylov), Ptr{Cvoid}, ())
        ptr == C_NULL && error("Failed to create Sector")
        
        obj = new(ptr)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall((:qkrylov_sector_destroy, libqkrylov), Cvoid, (Ptr{Cvoid},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end

    function Sector(ptr::Ptr{Cvoid})
        ptr == C_NULL && error("Invalid null Sector pointer")
        obj = new(ptr)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall((:qkrylov_sector_destroy, libqkrylov), Cvoid, (Ptr{Cvoid},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

function set_sz!(sec::Sector, sz2::Integer)
    status = ccall((:qkrylov_sector_set_sz, libqkrylov), Cint, (Ptr{Cvoid}, Cint), sec.ptr, Cint(sz2))
    status != QKRYLOV_SUCCESS && error("Failed to set sz sector (status code $status)")
    return sec
end

function set_hubbard_particles!(sec::Sector, nup::Integer, ndn::Integer)
    status = ccall((:qkrylov_sector_set_hubbard_particles, libqkrylov), Cint, (Ptr{Cvoid}, Cint, Cint), sec.ptr, Cint(nup), Cint(ndn))
    status != QKRYLOV_SUCCESS && error("Failed to set hubbard particles sector (status code $status)")
    return sec
end

function set_n!(sec::Sector, n::Integer)
    status = ccall((:qkrylov_sector_set_n, libqkrylov), Cint, (Ptr{Cvoid}, Cint), sec.ptr, Cint(n))
    status != QKRYLOV_SUCCESS && error("Failed to set n sector (status code $status)")
    return sec
end

function set_nb!(sec::Sector, nb::Integer)
    status = ccall((:qkrylov_sector_set_nb, libqkrylov), Cint, (Ptr{Cvoid}, Cint), sec.ptr, Cint(nb))
    status != QKRYLOV_SUCCESS && error("Failed to set nb sector (status code $status)")
    return sec
end

function get_sz(sec::Sector)::Union{Int, Nothing}
    sz2 = Ref{Cint}(0)
    active = Ref{Cint}(0)
    status = ccall((:qkrylov_sector_get_sz, libqkrylov), Cint, (Ptr{Cvoid}, Ref{Cint}, Ref{Cint}), sec.ptr, sz2, active)
    status != QKRYLOV_SUCCESS && error("Failed to query sz sector (status code $status)")
    return active[] != 0 ? Int(sz2[]) : nothing
end

function get_hubbard_particles(sec::Sector)::Union{Tuple{Int, Int}, Nothing}
    nup = Ref{Cint}(0)
    ndn = Ref{Cint}(0)
    active = Ref{Cint}(0)
    status = ccall((:qkrylov_sector_get_hubbard_particles, libqkrylov), Cint, (Ptr{Cvoid}, Ref{Cint}, Ref{Cint}, Ref{Cint}), sec.ptr, nup, ndn, active)
    status != QKRYLOV_SUCCESS && error("Failed to query hubbard particles sector (status code $status)")
    return active[] != 0 ? (Int(nup[]), Int(ndn[])) : nothing
end

function get_n(sec::Sector)::Union{Int, Nothing}
    n = Ref{Cint}(0)
    active = Ref{Cint}(0)
    status = ccall((:qkrylov_sector_get_n, libqkrylov), Cint, (Ptr{Cvoid}, Ref{Cint}, Ref{Cint}), sec.ptr, n, active)
    status != QKRYLOV_SUCCESS && error("Failed to query n sector (status code $status)")
    return active[] != 0 ? Int(n[]) : nothing
end

function get_nb(sec::Sector)::Union{Int, Nothing}
    nb = Ref{Cint}(0)
    active = Ref{Cint}(0)
    status = ccall((:qkrylov_sector_get_nb, libqkrylov), Cint, (Ptr{Cvoid}, Ref{Cint}, Ref{Cint}), sec.ptr, nb, active)
    status != QKRYLOV_SUCCESS && error("Failed to query nb sector (status code $status)")
    return active[] != 0 ? Int(nb[]) : nothing
end

function Base.show(io::IO, sec::Sector)
    constraints = String[]
    sz = get_sz(sec)
    sz !== nothing && push!(constraints, "2*Sz = $sz")
    hp = get_hubbard_particles(sec)
    hp !== nothing && push!(constraints, "N_up = $(hp[1]), N_dn = $(hp[2])")
    n = get_n(sec)
    n !== nothing && push!(constraints, "N = $n")
    nb = get_nb(sec)
    nb !== nothing && push!(constraints, "Nb = $nb")

    if isempty(constraints)
        print(io, "Sector(unconstrained)")
    else
        print(io, "Sector(", join(constraints, ", "), ")")
    end
end
