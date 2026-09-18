# OpSum term construction interface

struct OpTerm
    coeff::ComplexF64
    factors::Vector{Tuple{String, Int}}
end

struct OpExpr
    terms::Vector{OpTerm}
end

mutable struct OpSum
    ptr::Ptr{Cvoid}
    terms::Vector{OpTerm}

    function OpSum()
        ptr = ccall((:qkrylov_opsum_create, libqkrylov), Ptr{Cvoid}, ())
        ptr == C_NULL && error("Failed to create OpSum")
        obj = new(ptr, OpTerm[])
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall((:qkrylov_opsum_destroy, libqkrylov), Cvoid, (Ptr{Cvoid},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

Base.length(op::OpSum) = length(op.terms)
Base.size(op::OpSum) = (length(op.terms),)
Base.isempty(op::OpSum) = isempty(op.terms)
Base.getindex(op::OpSum, i::Integer) = op.terms[i]

function opsum_size(op::OpSum)::Int
    n = Ref{Cint}(0)
    status = ccall((:qkrylov_opsum_size, libqkrylov), Cint, (Ptr{Cvoid}, Ref{Cint}), op.ptr, n)
    _check_status(status, "Failed to query OpSum size")
    return Int(n[])
end

function opsum_get_term_info(op::OpSum, term_idx::Integer)
    re = Ref{Cdouble}(0.0)
    im = Ref{Cdouble}(0.0)
    n_fac = Ref{Cint}(0)
    status = ccall(
        (:qkrylov_opsum_get_term_info, libqkrylov),
        Cint,
        (Ptr{Cvoid}, Cint, Ref{Cdouble}, Ref{Cdouble}, Ref{Cint}),
        op.ptr, Cint(term_idx), re, im, n_fac
    )
    _check_status(status, "Failed to query OpSum term info")
    return (ComplexF64(re[], im[]), Int(n_fac[]))
end

function opsum_get_factor(op::OpSum, term_idx::Integer, factor_idx::Integer)
    buf = Vector{UInt8}(undef, 32)
    site = Ref{Cint}(0)
    status = ccall(
        (:qkrylov_opsum_get_factor, libqkrylov),
        Cint,
        (Ptr{Cvoid}, Cint, Cint, Ptr{Cchar}, Cint, Ref{Cint}),
        op.ptr, Cint(term_idx), Cint(factor_idx), pointer(buf), Cint(length(buf)), site
    )
    _check_status(status, "Failed to query OpSum factor")
    op_name = unsafe_string(pointer(buf))
    return (op_name, Int(site[]))
end

function clear!(op::OpSum)
    status = ccall((:qkrylov_opsum_clear, libqkrylov), Cint, (Ptr{Cvoid},), op.ptr)
    status != QKRYLOV_SUCCESS && error("Failed to clear OpSum (status code $status)")
    empty!(op.terms)
    return op
end

function add_term!(op::OpSum, coeff::Number, op1::AbstractString, site1::Integer)
    c = ComplexF64(coeff)
    status = ccall(
        (:qkrylov_opsum_add_term_1body_fp64, libqkrylov),
        Cint,
        (Ptr{Cvoid}, Cdouble, Cdouble, Cstring, Cint),
        op.ptr, Cdouble(real(c)), Cdouble(imag(c)), string(op1), Cint(site1)
    )
    status != QKRYLOV_SUCCESS && error("Failed to add 1-body term to OpSum (status code $status)")
    push!(op.terms, OpTerm(c, [(string(op1), Int(site1))]))
    return op
end

function add_term!(op::OpSum, coeff::Number, op1::AbstractString, site1::Integer, op2::AbstractString, site2::Integer)
    c = ComplexF64(coeff)
    status = ccall(
        (:qkrylov_opsum_add_term_2body_fp64, libqkrylov),
        Cint,
        (Ptr{Cvoid}, Cdouble, Cdouble, Cstring, Cint, Cstring, Cint),
        op.ptr, Cdouble(real(c)), Cdouble(imag(c)), string(op1), Cint(site1), string(op2), Cint(site2)
    )
    status != QKRYLOV_SUCCESS && error("Failed to add 2-body term to OpSum (status code $status)")
    push!(op.terms, OpTerm(c, [(string(op1), Int(site1)), (string(op2), Int(site2))]))
    return op
end

function add_term!(op::OpSum, coeff::Number, ops::AbstractVector{<:AbstractString}, sites::AbstractVector{<:Integer})
    @assert length(ops) == length(sites) "Length of operators ($(length(ops))) does not match length of site indices ($(length(sites)))"
    n_factors = length(ops)
    if n_factors == 1
        return add_term!(op, coeff, ops[1], sites[1])
    elseif n_factors == 2
        return add_term!(op, coeff, ops[1], sites[1], ops[2], sites[2])
    end

    c = ComplexF64(coeff)
    c_ops = [string(o) for o in ops]
    c_sites = Cint[Cint(s) for s in sites]
    ops_ptrs = Ptr{Cchar}[pointer(s) for s in c_ops]

    GC.@preserve c_ops c_sites ops_ptrs begin
        status = ccall(
            (:qkrylov_opsum_add_term_nbody_fp64, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Cdouble, Cdouble, Cint, Ptr{Ptr{Cchar}}, Ptr{Cint}),
            op.ptr, Cdouble(real(c)), Cdouble(imag(c)), Cint(n_factors), ops_ptrs, c_sites
        )
    end
    status != QKRYLOV_SUCCESS && error("Failed to add $n_factors-body term to OpSum (status code $status)")
    push!(op.terms, OpTerm(c, [(string(ops[k]), Int(sites[k])) for k in 1:n_factors]))
    return op
end

# -----------------------------------------------------------------------------
# Operator Term Arithmetic & Generator Functions (Sz, Sp, Sm, Sx, Sy, n, c, cdag)
# -----------------------------------------------------------------------------

# Spin-1/2 Site Operator Generators
Sz(site::Integer)   = OpTerm(1.0, [("Sz", Int(site))])
Sp(site::Integer)   = OpTerm(1.0, [("Sp", Int(site))])
Sm(site::Integer)   = OpTerm(1.0, [("Sm", Int(site))])
Sx(site::Integer)   = OpTerm(1.0, [("Sx", Int(site))])
Sy(site::Integer)   = OpTerm(1.0, [("Sy", Int(site))])

# Spinless Fermion Operator Generators
n(site::Integer)    = OpTerm(1.0, [("n",    Int(site))])
c(site::Integer)    = OpTerm(1.0, [("c",    Int(site))])
cdag(site::Integer) = OpTerm(1.0, [("cdag", Int(site))])

# Hubbard / Spinful Electron Operator Generators
CdagUp(site::Integer) = OpTerm(1.0, [("CdagUp", Int(site))])
CUp(site::Integer)    = OpTerm(1.0, [("CUp",    Int(site))])
CdagDn(site::Integer) = OpTerm(1.0, [("CdagDn", Int(site))])
CDn(site::Integer)    = OpTerm(1.0, [("CDn",    Int(site))])
Nup(site::Integer)    = OpTerm(1.0, [("Nup",    Int(site))])
Ndn(site::Integer)    = OpTerm(1.0, [("Ndn",    Int(site))])
Nupdn(site::Integer)  = OpTerm(1.0, [("Nupdn",  Int(site))])

# Boson Operator Generators
Bdag(site::Integer)   = OpTerm(1.0, [("Bdag",   Int(site))])
B(site::Integer)      = OpTerm(1.0, [("B",      Int(site))])
N(site::Integer)      = OpTerm(1.0, [("N",      Int(site))])


# Scaling (*)
Base.:*(a::Number, t::OpTerm) = OpTerm(ComplexF64(a) * t.coeff, t.factors)
Base.:*(t::OpTerm, a::Number) = OpTerm(t.coeff * ComplexF64(a), t.factors)

Base.:*(a::Number, expr::OpExpr) = OpExpr([a * t for t in expr.terms])
Base.:*(expr::OpExpr, a::Number) = OpExpr([t * a for t in expr.terms])

# Unary minus (-)
Base.:-(t::OpTerm) = OpTerm(-t.coeff, t.factors)
Base.:-(expr::OpExpr) = OpExpr([-t for t in expr.terms])

# Term multiplication (*)
Base.:*(t1::OpTerm, t2::OpTerm) = OpTerm(t1.coeff * t2.coeff, vcat(t1.factors, t2.factors))

# Addition (+)
Base.:+(t1::OpTerm, t2::OpTerm) = OpExpr([t1, t2])
Base.:+(expr::OpExpr, t::OpTerm) = OpExpr(vcat(expr.terms, [t]))
Base.:+(t::OpTerm, expr::OpExpr) = OpExpr(vcat([t], expr.terms))
Base.:+(e1::OpExpr, e2::OpExpr)   = OpExpr(vcat(e1.terms, e2.terms))

# Subtraction (-)
Base.:-(t1::OpTerm, t2::OpTerm) = t1 + (-t2)
Base.:-(expr::OpExpr, t::OpTerm) = expr + (-t)
Base.:-(t::OpTerm, expr::OpExpr) = t + (-expr)
Base.:-(e1::OpExpr, e2::OpExpr)   = e1 + (-e2)

# Adding terms/expressions into OpSum
function add_term!(op::OpSum, t::OpTerm)
    ops = String[f[1] for f in t.factors]
    sites = Int[f[2] for f in t.factors]
    add_term!(op, t.coeff, ops, sites)
    return op
end

function add_term!(op::OpSum, expr::OpExpr)
    for t in expr.terms
        add_term!(op, t)
    end
    return op
end

# OpSum arithmetic (os += expr, os -= expr)
Base.:+(op::OpSum, t::OpTerm) = add_term!(op, t)
Base.:+(op::OpSum, expr::OpExpr) = add_term!(op, expr)
Base.:-(op::OpSum, t::OpTerm) = add_term!(op, -t)
Base.:-(op::OpSum, expr::OpExpr) = add_term!(op, -expr)

# -----------------------------------------------------------------------------
# Base.show Formatting and Validation
# -----------------------------------------------------------------------------

function Base.show(io::IO, t::OpTerm)
    coeff_str = if imag(t.coeff) == 0
        isinteger(real(t.coeff)) ? string(Int(real(t.coeff))) : string(real(t.coeff))
    else
        string(t.coeff)
    end
    factors_str = join(["$(f[1])($(f[2]))" for f in t.factors], " * ")
    if coeff_str == "1"
        print(io, factors_str)
    elseif coeff_str == "-1"
        print(io, "-$factors_str")
    else
        print(io, "$coeff_str * $factors_str")
    end
end

function Base.show(io::IO, op::OpSum)
    n = length(op)
    if n == 0
        print(io, "OpSum(empty)")
    elseif n <= 5
        print(io, "OpSum(", join([string(t) for t in op.terms], " + "), ")")
    else
        print(io, "OpSum($n terms: ", join([string(t) for t in op.terms[1:3]], " + "), " ...)")
    end
end

function Base.show(io::IO, ::MIME"text/plain", op::OpSum)
    n = length(op)
    if n == 0
        println(io, "OpSum with 0 terms (empty)")
    else
        println(io, "OpSum with $n terms:")
        for (idx, t) in enumerate(op.terms)
            println(io, "  [$idx] $t")
        end
    end
end

function validate(op::OpSum, num_sites::Integer)
    errors = String[]
    for (idx, t) in enumerate(op.terms)
        if isnan(t.coeff) || isinf(t.coeff)
            push!(errors, "Term $idx has non-finite coefficient: $(t.coeff)")
        end
        for (op_name, site) in t.factors
            if site < 0 || site >= num_sites
                push!(errors, "Term $idx: site index $site for operator '$op_name' is out of bounds for a $num_sites-site system (valid 0-indexed range: 0..$(num_sites - 1))")
            end
        end
    end
    return (isempty(errors), errors)
end

function validate!(op::OpSum, num_sites::Integer)
    valid, errors = validate(op, num_sites)
    if !valid
        error_msg = join(errors, "\n  - ")
        throw(ArgumentError("OpSum validation failed:\n  - $error_msg"))
    end
    return op
end
