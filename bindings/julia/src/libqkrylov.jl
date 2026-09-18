# Shared library loader and C API status codes for qkrylov

const QKRYLOV_SUCCESS           =  0
const QKRYLOV_ERROR_INVALID_ARG = -1
const QKRYLOV_ERROR_EXCEPTION   = -2

struct LocalActionC
    valid::Cint
    new_state::UInt64
    matrix_element_re::Cdouble
    matrix_element_im::Cdouble
end

@enum BasisTypeC begin
    BASIS_SPIN_HALF = 0
    BASIS_SPIN_S    = 1
    BASIS_FERMION   = 2
    BASIS_HUBBARD   = 3
    BASIS_TJ        = 4
end

@enum SiteTypeC begin
    SITE_SPIN_HALF = 0
    SITE_SPIN_S    = 1
    SITE_FERMION   = 2
    SITE_HUBBARD   = 3
    SITE_TJ        = 4
end

"""
    get_last_error_message() -> String

Returns the last error message caught across the C ABI barrier on the current calling thread.
Returns an empty string `""` if no error occurred.
"""
function get_last_error_message()::String
    ptr = ccall((:qkrylov_get_last_error_message, libqkrylov), Cstring, ())
    return ptr == C_NULL ? "" : unsafe_string(ptr)
end

"""
    clear_last_error()

Clears the thread-local error message for the current calling thread.
"""
function clear_last_error()
    ccall((:qkrylov_clear_last_error, libqkrylov), Cvoid, ())
end

function _check_status(status::Integer, default_msg::AbstractString)
    if status != QKRYLOV_SUCCESS
        err = get_last_error_message()
        if !isempty(err)
            error("$default_msg (status code $status): $err")
        else
            error("$default_msg with status code $status")
        end
    end
end

function find_libqkrylov()
    # 1. Custom environment variable override (for local development or CI)
    if haskey(ENV, "QKRYLOV_LIB_PATH")
        p = ENV["QKRYLOV_LIB_PATH"]
        if isfile(p)
            return p
        else
            @warn "QKRYLOV_LIB_PATH is set to '$p' but file was not found. Searching fallback paths."
        end
    end

    # 2. Local repository build path (for development)
    root_dir = normpath(joinpath(@__DIR__, "..", "..", ".."))
    candidates = [
        joinpath(root_dir, "build", "libqkrylov.so"),
        joinpath(root_dir, "build", "libqkrylov.dylib"),
        joinpath(root_dir, "build", "qkrylov.dll"),
        joinpath(root_dir, "build", "Release", "libqkrylov.so"),
        joinpath(root_dir, "build", "Release", "libqkrylov.dylib"),
        joinpath(root_dir, "build", "Release", "qkrylov.dll"),
        joinpath(root_dir, "build", "Debug", "libqkrylov.so"),
        joinpath(root_dir, "build", "Debug", "libqkrylov.dylib"),
        joinpath(root_dir, "build", "Debug", "qkrylov.dll")
    ]

    for path in candidates
        if isfile(path)
            return path
        end
    end

    # 3. Native Julia Artifacts resolution (auto-downloaded by Pkg via Artifacts.toml)
    artifacts_file = joinpath(@__DIR__, "..", "Artifacts.toml")
    if isfile(artifacts_file)
        try
            # Augment platform for CUDA resolution (matching Yggdrasil CUDA.augment)
            platform = deepcopy(Base.BinaryPlatforms.HostPlatform())
            has_cuda = false
            try
                h = Libdl.dlopen("libcuda"; throw_error=false)
                if h !== nothing
                    sym = Libdl.dlsym(h, :cuDriverGetVersion; throw_error=false)
                    if sym !== nothing
                        ver = Ref{Cint}(0)
                        ccall(sym, Cint, (Ptr{Cint},), ver)
                        if div(ver[], 1000) >= 12
                            platform["cuda"] = "12.0"
                            has_cuda = true
                        end
                    end
                end
            catch
            end
            if !has_cuda && Sys.islinux() && Sys.ARCH === :x86_64
                platform["cuda"] = "none"
            end

            meta = Artifacts.artifact_meta("libqkrylov", artifacts_file; platform=platform)
            # If CUDA artifact not found or not built yet, fallback to CPU
            if meta === nothing && has_cuda
                platform["cuda"] = "none"
                meta = Artifacts.artifact_meta("libqkrylov", artifacts_file; platform=platform)
            end

            if meta !== nothing
                hash = Base.SHA1(meta["git-tree-sha1"])
                if !Artifacts.artifact_exists(hash)
                    try
                        pkg_mod = Base.require(Base.PkgId(Base.UUID("44cfe95a-1eb2-52ea-b672-e2afdf69b78f"), "Pkg"))
                        Base.invokelatest(getfield(getfield(pkg_mod, :Artifacts), :ensure_artifact_installed), "libqkrylov", artifacts_file; platform=platform)
                    catch
                    end
                end
                if Artifacts.artifact_exists(hash)
                    artifact_dir = Artifacts.artifact_path(hash)
                    for candidate_rel in [
                        joinpath("lib", "libqkrylov.so"),
                        joinpath("lib64", "libqkrylov.so"),
                        joinpath("lib", "libqkrylov.dylib"),
                        joinpath("lib", "qkrylov.dll"),
                        joinpath("bin", "qkrylov.dll"),
                        "libqkrylov.so",
                        "libqkrylov.dylib",
                        "qkrylov.dll"
                    ]
                        candidate_path = joinpath(artifact_dir, candidate_rel)
                        if isfile(candidate_path)
                            if has_cuda
                                h = Libdl.dlopen(candidate_path; throw_error=false)
                                if h !== nothing
                                    Libdl.dlclose(h)
                                    return candidate_path
                                else
                                    break
                                end
                            else
                                return candidate_path
                            end
                        end
                    end
                end
            end

            if has_cuda
                platform["cuda"] = "none"
                meta_cpu = Artifacts.artifact_meta("libqkrylov", artifacts_file; platform=platform)
                if meta_cpu !== nothing
                    hash_cpu = Base.SHA1(meta_cpu["git-tree-sha1"])
                    if !Artifacts.artifact_exists(hash_cpu)
                        try
                            pkg_mod = Base.require(Base.PkgId(Base.UUID("44cfe95a-1eb2-52ea-b672-e2afdf69b78f"), "Pkg"))
                            Base.invokelatest(getfield(getfield(pkg_mod, :Artifacts), :ensure_artifact_installed), "libqkrylov", artifacts_file; platform=platform)
                        catch
                        end
                    end
                    if Artifacts.artifact_exists(hash_cpu)
                        cpu_dir = Artifacts.artifact_path(hash_cpu)
                        for cand in [joinpath("lib", "libqkrylov.so"), joinpath("lib64", "libqkrylov.so"), "libqkrylov.so"]
                            p = joinpath(cpu_dir, cand)
                            if isfile(p)
                                return p
                            end
                        end
                    end
                end
            end
        catch
        end
    end

    # 4. Fallback to qkrylov_jll if available in runtime environment
    try
        if isdefined(QuantumKrylov, :qkrylov_jll) && isdefined(qkrylov_jll, :libqkrylov)
            return qkrylov_jll.libqkrylov
        end
    catch
    end

    # 5. Fallback to system library resolution
    return "libqkrylov"
end

global libqkrylov::String = find_libqkrylov()

function __init__()
    global libqkrylov = find_libqkrylov()
end

