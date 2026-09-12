set(QKRYLOV_TESTS
    test_hamiltonian_sz
    test_heisenberg_2site
    test_lanczos_2site
    test_opsum
    test_spinhalf_basis
    test_spinhalf_site
    test_vector_ops
    test_fermion_basis
    test_fermion_site
    test_davidson
    test_lanczos_ritz
    test_dynamics
    test_tj_basis
    test_tj_site
    test_ftlm
    test_workflow
    test_c_api
    test_spin1_correction_vector
    test_m1_core
    test_lanczos_lowest
    test_lanczos_policies
    test_ftlm_sweep
)

foreach(test_name IN LISTS QKRYLOV_TESTS)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE qkrylov::qkrylov)
    target_compile_definitions(${test_name} PRIVATE QKRYLOV_DOUBLE_PRECISION)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# FP32 unit tests to ensure solvers and bases deflate and converge under single precision
set(QKRYLOV_FP32_TESTS
    test_lanczos_2site
    test_lanczos_ritz
    test_davidson
    test_dynamics
    test_ftlm
    test_m1_core
    test_lanczos_policies
    test_ftlm_sweep
)

foreach(test_name IN LISTS QKRYLOV_FP32_TESTS)
    set(target_name "${test_name}_fp32")
    add_executable(${target_name} ${test_name}.cpp)
    target_link_libraries(${target_name} PRIVATE qkrylov::qkrylov)
    target_compile_definitions(${target_name} PRIVATE QKRYLOV_SINGLE_PRECISION)
    add_test(NAME ${target_name} COMMAND ${target_name})
endforeach()

