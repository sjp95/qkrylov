set(QKRYLOV_TESTS
    test_hamiltonian_sz
    test_heisenberg_2site
    test_lanczos_2site
    test_opsum
    test_spinhalf_basis
    test_spinhalf_site
    test_vector_ops
)

foreach(test_name IN LISTS QKRYLOV_TESTS)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE qkrylov::qkrylov)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

