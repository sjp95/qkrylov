cmake_minimum_required(VERSION 3.20)

if(DEFINED SKBUILD_PROJECT_VERSION)
    set(QKRYLOV_PROJECT_VERSION "${SKBUILD_PROJECT_VERSION}")
else()
    set(QKRYLOV_PROJECT_VERSION "0.0.0")
endif()

project(
    qkrylov
    VERSION ${QKRYLOV_PROJECT_VERSION}
    DESCRIPTION "Matrix-free Krylov methods for quantum many-body physics"
    LANGUAGES CXX
)

option(QKRYLOV_BUILD_TESTS "Build qkrylov test executables" ON)
option(QKRYLOV_BUILD_EXAMPLES "Build qkrylov examples" OFF)
option(QKRYLOV_BUILD_PYTHON "Build Python bindings" OFF)

set(QKRYLOV_SOURCES
    src/basis/spinhalf_basis.cpp
    src/basis/spin_s_basis.cpp
    src/basis/fermion_basis.cpp
    src/basis/hubbard_basis.cpp
    src/basis/tj_basis.cpp
    src/hamiltonian/matrix_free_hamiltonian.cpp
    src/operators/opsum.cpp
    src/sites/spinhalf_site.cpp
    src/sites/spin_s_site.cpp
    src/sites/fermion_site.cpp
    src/sites/hubbard_site.cpp
    src/sites/tj_site.cpp
    src/solvers/lanczos.cpp
    src/solvers/davidson.cpp
    src/solvers/dynamics.cpp
    src/solvers/correction_vector.cpp
    src/solvers/ftlm.cpp
)

add_library(qkrylov_fp32 OBJECT ${QKRYLOV_SOURCES})
target_compile_definitions(qkrylov_fp32 PUBLIC QKRYLOV_SINGLE_PRECISION)
target_compile_features(qkrylov_fp32 PUBLIC cxx_std_20)
set_target_properties(qkrylov_fp32 PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(qkrylov_fp64 OBJECT ${QKRYLOV_SOURCES})
target_compile_definitions(qkrylov_fp64 PUBLIC QKRYLOV_DOUBLE_PRECISION)
target_compile_features(qkrylov_fp64 PUBLIC cxx_std_20)
set_target_properties(qkrylov_fp64 PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(qkrylov $<TARGET_OBJECTS:qkrylov_fp32> $<TARGET_OBJECTS:qkrylov_fp64> src/c_api.cpp)
add_library(qkrylov::qkrylov ALIAS qkrylov)
target_compile_features(qkrylov PUBLIC cxx_std_20)
set_target_properties(qkrylov PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    target_compile_definitions(qkrylov_fp32 PUBLIC _USE_MATH_DEFINES)
    target_compile_definitions(qkrylov_fp64 PUBLIC _USE_MATH_DEFINES)
endif()

# find_package(Kokkos QUIET)
if(NOT Kokkos_FOUND)
    message(STATUS "Kokkos not found on system. Fetching Kokkos 4.3.01 from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        kokkos
        GIT_REPOSITORY https://github.com/kokkos/kokkos.git
        GIT_TAG        4.7.04
    )
    # Force Position Independent Code globally so Kokkos thread_local variables use global-dynamic TLS
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(Kokkos_ENABLE_PIC ON CACHE BOOL "Enable Position Independent Code for Kokkos" FORCE)
    
    # If no backend was explicitly passed by the user, default to OpenMP for CPU fallback
    if(NOT DEFINED Kokkos_ENABLE_CUDA AND NOT DEFINED Kokkos_ENABLE_HIP AND NOT DEFINED Kokkos_ENABLE_SYCL)
        if(MSVC)
            set(Kokkos_ENABLE_OPENMP OFF CACHE BOOL "Disable Kokkos OpenMP on MSVC" FORCE)
        else()
            set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable Kokkos OpenMP backend" FORCE)
        endif()
        set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Enable Kokkos Serial backend" FORCE)
    endif()
    # Suppress Kokkos warnings during FetchContent
    set(Kokkos_ENABLE_DEPRECATION_WARNINGS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(kokkos)
endif()

target_link_libraries(qkrylov_fp32 PUBLIC Kokkos::kokkos)
target_link_libraries(qkrylov_fp64 PUBLIC Kokkos::kokkos)
target_link_libraries(qkrylov PUBLIC Kokkos::kokkos)

target_include_directories(qkrylov_fp32 PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_include_directories(qkrylov_fp64 PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_include_directories(qkrylov PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)

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

if(NOT DEFINED SKBUILD_PROJECT_NAME)
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
endif()

if(QKRYLOV_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(QKRYLOV_BUILD_PYTHON)
    find_package(Python 3.11 REQUIRED
        COMPONENTS Interpreter Development.Module
        OPTIONAL_COMPONENTS Development.SABIModule
    )
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(
        _qkrylov_cpp
        STABLE_ABI
        NB_DOMAIN qkrylov
        bindings/python/src/binding_main.cpp bindings/python/src/binding_fp32.cpp bindings/python/src/binding_fp64.cpp
    )

    target_link_libraries(_qkrylov_cpp PRIVATE qkrylov::qkrylov)
    if(Kokkos_ENABLE_SYCL)
        target_link_options(_qkrylov_cpp PRIVATE -fsycl)
        target_compile_options(_qkrylov_cpp PRIVATE -fsycl)
    endif()

    install(
        TARGETS _qkrylov_cpp
        LIBRARY DESTINATION qkrylov
        RUNTIME DESTINATION qkrylov
    )

    install(
        DIRECTORY bindings/python/qkrylov/
        DESTINATION qkrylov
        FILES_MATCHING
            PATTERN "*.py"
            PATTERN "py.typed"
    )
endif()
