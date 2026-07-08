cmake_minimum_required(VERSION 3.20)

project(
    qkrylov
    VERSION 0.1.0
    DESCRIPTION "Matrix-free Krylov methods for quantum many-body physics"
    LANGUAGES CXX
)

option(QKRYLOV_BUILD_TESTS "Build qkrylov test executables" ON)
option(QKRYLOV_BUILD_EXAMPLES "Build qkrylov examples" OFF)
option(QKRYLOV_ENABLE_CUDA "Enable CUDA acceleration" OFF)

if(QKRYLOV_ENABLE_CUDA)
    enable_language(CUDA)
endif()

add_library(qkrylov
    src/basis/spinhalf_basis.cpp
    src/basis/fermion_basis.cpp
    src/basis/hubbard_basis.cpp
    src/basis/tj_basis.cpp
    src/basis/generic_basis.cpp
    src/hamiltonian/matrix_free_hamiltonian.cpp
    src/operators/opsum.cpp
    src/sites/spinhalf_site.cpp
    src/sites/fermion_site.cpp
    src/sites/hubbard_site.cpp
    src/sites/tj_site.cpp
    src/sites/boson_site.cpp
    src/sites/hardcore_boson_site.cpp
    src/sites/spin_s_site.cpp
    src/solvers/lanczos.cpp
    src/solvers/davidson.cpp
    src/solvers/dynamics.cpp
    src/solvers/ftlm.cpp
)

if(QKRYLOV_ENABLE_CUDA)
    target_sources(qkrylov PRIVATE src/hamiltonian/cuda_hamiltonian.cu)
    target_compile_definitions(qkrylov PUBLIC QKRYLOV_ENABLE_CUDA)
endif()

add_library(qkrylov::qkrylov ALIAS qkrylov)

target_compile_features(qkrylov PUBLIC cxx_std_20)
set_target_properties(qkrylov PROPERTIES POSITION_INDEPENDENT_CODE ON)

find_package(OpenMP REQUIRED)
target_link_libraries(qkrylov PUBLIC OpenMP::OpenMP_CXX)

target_include_directories(qkrylov
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

configure_package_config_file(
    cmake/qkrylovConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/qkrylovConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/qkrylov
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/qkrylovConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    TARGETS qkrylov
    EXPORT qkrylovTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    EXPORT qkrylovTargets
    NAMESPACE qkrylov::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/qkrylov
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/qkrylovConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/qkrylovConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/qkrylov
)

if(QKRYLOV_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
    qkrylov_cpp
    bindings/python/src/binding.cpp
)

target_link_libraries(qkrylov_cpp PRIVATE qkrylov::qkrylov)
