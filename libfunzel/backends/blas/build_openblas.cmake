
option(USE_OPENMP "Use OpenMP with OpenBLAS" TRUE)
option(USE_THREAD "Use multithreading with OpenBLAS" TRUE)

set(C_LAPACK TRUE CACHE BOOL "Use C sources, Fortran may not be available." FORCE)

set(INTERFACE64 FALSE CACHE BOOL "Use 64-bit BLAS interface.")
if(INTERFACE64)
	set(BUILD_INDEX64 TRUE CACHE BOOL "Use 64-bit indexing.")

	set(FUNZEL_BLAS_DEFINITIONS -DLAPACK_DISABLE_NAN_CHECK=1 -DLAPACK_ILP64=1 -DFUNZEL_ILP64=1)
	add_compile_definitions(-DLAPACK_DISABLE_NAN_CHECK=1 -DLAPACK_ILP64=1 -DFUNZEL_ILP64=1)
	set(UNDERSCORE64 "_64")
	set(POSTFIX64 "64")
endif()

if(MSVC)
	set(FUNZEL_BLAS_DEFINITIONS ${FUNZEL_BLAS_DEFINITIONS} -DHAVE_LAPACK_CONFIG_H -DLAPACK_COMPLEX_STRUCTURE)
	set(DYNAMIC_ARCH FALSE CACHE BOOL "Use dynamic arch for flexibility.")
else()
	set(DYNAMIC_ARCH TRUE CACHE BOOL "Use dynamic arch for flexibility.")
endif()

set(NO_WARMUP TRUE CACHE BOOL "Prevent expensive warmup.")
set(DYNAMIC_LIST "HASWELL;ZEN;SKYLAKEX" CACHE STRING "List of optimized BLAS arches.")

if(FUNZEL_USE_SHARED_BLAS)
	set(BUILD_SHARED_LIBS TRUE)
	set(BUILD_STATIC_LIBS FALSE)
	set(FUNZEL_BLAS_LIBRARIES openblas${UNDERSCORE64}_shared)
else()
	set(BUILD_STATIC_LIBS TRUE)
	set(FUNZEL_BLAS_LIBRARIES openblas${UNDERSCORE64}_static)
endif()

add_compile_definitions(${FUNZEL_BLAS_DEFINITIONS})

## OpenBLAS does not support LTO!
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE FALSE)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO FALSE)

add_subdirectory(openblas EXCLUDE_FROM_ALL)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ${FUNZEL_ENABLE_LTO})
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ${FUNZEL_ENABLE_LTO})

set(FUNZEL_BLAS_INCLUDE_DIRS
	"${CMAKE_BINARY_DIR}/generated"
	"${CMAKE_BINARY_DIR}"
	"openblas"
	"openblas/lapack-netlib/LAPACKE/include/"
	"${CMAKE_CURRENT_BINARY_DIR}/openblas")

set(BLA_VENDOR "OpenBLAS")

if(MSVC AND FUNZEL_USE_SHARED_BLAS) ## OpenBLAS outputs somewhere else, fix that.
	add_custom_command(TARGET funzelBlas POST_BUILD
	COMMAND
		${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/openblas/lib/$<$<CONFIG:Debug>:Debug>$<$<CONFIG:Release>:Release>$<$<CONFIG:RelWithDebInfo>:RelWithDebInfo>/openblas${UNDERSCORE64}.dll ${CMAKE_BINARY_DIR}/bin/$<$<CONFIG:Debug>:Debug>$<$<CONFIG:Release>:Release>$<$<CONFIG:RelWithDebInfo>:RelWithDebInfo>/
	)
endif()
