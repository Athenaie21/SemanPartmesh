cmake_minimum_required(VERSION 3.16)
project(QuadExtract CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

# ---- Eigen ----
find_package(Eigen3 REQUIRED)

# ---- OpenMesh (needed by libQEx header) ----
find_package(OpenMesh REQUIRED)

# ---- BLAS / LAPACK (CoMISo dependency) ----
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# ---- SuiteSparse (CoMISo uses cholmod/umfpack) ----
find_package(PkgConfig QUIET)
pkg_check_modules(SUITESPARSE QUIET IMPORTED_TARGET SuiteSparse)

# ---- Paths ----
set(LIBIGL_DIR    ${ROOT}/libigl)
set(COMISO_SRC    ${LIBIGL_DIR}/CoMISo)
set(COMISO_BUILD  ${COMISO_SRC}/build)
set(LIBQEX_DIR    ${ROOT}/libQEx)

# ---- CoMISo static library ----
add_library(CoMISo_imported STATIC IMPORTED)
set_target_properties(CoMISo_imported PROPERTIES
    IMPORTED_LOCATION ${COMISO_BUILD}/Build/lib/libCoMISod.a
)

# ---- libQEx static library ----
add_library(QEx_imported STATIC IMPORTED)
set_target_properties(QEx_imported PROPERTIES
    IMPORTED_LOCATION ${LIBQEX_DIR}/build/libQExStatic.a
)

# ---- Executable ----
add_executable(extract_quad_mesh extract_quad_mesh.cpp)

target_include_directories(extract_quad_mesh PRIVATE
    # libigl (header-only)
    ${LIBIGL_DIR}/include

    # CoMISo headers
    ${COMISO_SRC}/src
    ${COMISO_SRC}/src/CoMISo
    ${COMISO_SRC}/include
    ${COMISO_BUILD}/include
    ${COMISO_BUILD}/include/CoMISo

    # GMM (system)
    /usr/include

    # SuiteSparse (cholmod, umfpack, etc.)
    /usr/include/suitesparse

    # libQEx headers
    ${LIBQEX_DIR}/interfaces/c
)

find_library(OPENMESH_CORE OpenMeshCore PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib)
find_library(OPENMESH_TOOLS OpenMeshTools PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib)

target_link_libraries(extract_quad_mesh PRIVATE
    QEx_imported
    CoMISo_imported
    Eigen3::Eigen
    ${OPENMESH_CORE}
    ${OPENMESH_TOOLS}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

if(SUITESPARSE_FOUND)
    target_link_libraries(extract_quad_mesh PRIVATE PkgConfig::SUITESPARSE)
else()
    find_library(CHOLMOD_LIB cholmod)
    find_library(UMFPACK_LIB umfpack)
    find_library(AMD_LIB amd)
    find_library(COLAMD_LIB colamd)
    find_library(SUITESPARSE_CONFIG suitesparseconfig)
    target_link_libraries(extract_quad_mesh PRIVATE
        ${CHOLMOD_LIB} ${UMFPACK_LIB} ${AMD_LIB} ${COLAMD_LIB} ${SUITESPARSE_CONFIG}
    )
endif()
