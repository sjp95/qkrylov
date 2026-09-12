module QuantumKrylov

using qkrylov_jll

const VERSION = v"0.1.0"

include("libqkrylov.jl")
include("device.jl")
include("sector.jl")
include("site.jl")
include("basis.jl")
include("opsum.jl")
include("hamiltonian.jl")
include("solvers.jl")
include("vector_ops.jl")

export Sector, set_sz!, set_hubbard_particles!, set_n!, set_nb!, get_sz, get_hubbard_particles, get_n, get_nb
export AbstractSite, SpinHalfSite, SpinSSite, FermionSite, HubbardSite, TJSite, LocalAction, apply, site_type
export AbstractBasis, SpinHalfBasis, SpinSBasis, FermionBasis, HubbardBasis, TJBasis, dimension, nsites, state, basis_index, basis_type, sector
export spin, dimension_per_site
export OpSum, add_term!, clear!, OpTerm, OpExpr, opsum_size, opsum_get_term_info, opsum_get_factor
export Sz, Sp, Sm, Sx, Sy, n, c, cdag
export CdagUp, CUp, CdagDn, CDn, Nup, Ndn, Nupdn, Bdag, B, N
export validate, validate!
export MatrixFreeHamiltonian, diagonal, diagonal_device
export DeviceVector, mul!
export solve
export AbstractQuantumProblem, GroundStateProblem, ExcitedStatesProblem, ThermalProblem, DynamicsProblem, SpectralProblem
export AbstractQuantumAlgorithm, AbstractLanczosVariation, SinglePass, TwoPass
export Lanczos, Davidson, FTLM, ContinuedFraction, CorrectionVector
export AbstractQuantumSolution, GroundStateSolution, LanczosResult, ExcitedStatesSolution
export lanczos_ground_state, lanczos_lowest, LanczosLowestResult
export davidson_lowest, DavidsonResult
export continued_fraction_coeffs, ContinuedFractionResult, evaluate_spectral_function
export ftlm, FTLMResult, ftlm_sweep, FTLMSweepResult
export solver_correction_vector, CorrectionVectorResult
export vector_dot, vector_norm, vector_axpy!, vector_scal!, vector_normalize!, vector_zero_fill!, vector_copy!
export AbstractDevice, CPUDevice, CUDADevice, HIPDevice, SYCLDevice
export find_gpu, gpu_count, is_gpu_build, initialize_device!
export get_last_error_message, clear_last_error

end
