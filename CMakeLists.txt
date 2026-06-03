cmake_minimum_required(VERSION 3.20)

project(
    qkrylov
    VERSION 0.1.0
    DESCRIPTION "Matrix-free Krylov methods for quantum many-body physics"
    LANGUAGES CXX
)

option(QKRYLOV_BUILD_TESTS "Build qkrylov test executables" ON)
option(QKRYLOV_BUILD_EXAMPLES "Build qkrylov examples" OFF)

add_library(qkrylov
    src/basis/spinhalf_basis.cpp
    src/hamiltonian/matrix_free_hamiltonian.cpp
    src/operators/opsum.cpp
    src/sites/spinhalf_site.cpp
    src/solvers/lanczos.cpp
)

add_library(qkrylov::qkrylov ALIAS qkrylov)

target_compile_features(qkrylov PUBLIC cxx_std_20)

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
