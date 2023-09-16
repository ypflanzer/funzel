file(GLOB SRC src/*.cpp)
file(GLOB HEADERS src/*.h*)

option(USE_INTERNAL_BLAS "Compile OpenBLAS ourselves instead of relying on system packages." ON)
if(USE_INTERNAL_BLAS)

	set(C_LAPACK TRUE CACHE BOOL "Use C sources, Fortran may not be available." FORCE)
	set(DYNAMIC_ARCH TRUE CACHE BOOL "Use dynamic arch for flexibility.")
	set(NO_WARMUP TRUE CACHE BOOL "Prevent expensive warmup.")
	set(DYNAMIC_LIST "HASWELL;ZEN;SKYLAKEX" CACHE STRING "List of optimized BLAS arches.")

	if(WIN32)
		set(BUILD_STATIC_LIBS FALSE CACHE BOOL "Build static libs." FORCE)
		set(FUNZEL_BLAS_LIBRARIES openblas_shared)
	else()
		set(BUILD_STATIC_LIBS TRUE CACHE BOOL "Build static libs." FORCE)
		set(FUNZEL_BLAS_LIBRARIES openblas_static)
	endif()

	add_subdirectory(openblas EXCLUDE_FROM_ALL)
	set(FUNZEL_BLAS_INCLUDE_DIRS
		"openblas"
		"openblas/lapack-netlib/LAPACKE/include/"
		"${CMAKE_CURRENT_BINARY_DIR}/openblas")

	set(BLA_VENDOR "OpenBLAS")
else()
	if(APPLE)
		set(CMAKE_PREFIX_PATH "/usr/local/opt/openblas")
	endif()

	set(BLA_VENDOR "OpenBLAS" CACHE STRING "The BLAS implementation to be used.")
	find_package(BLAS REQUIRED)
	find_package(LAPACK REQUIRED)

	set(FUNZEL_BLAS_LIBRARIES BLAS::BLAS ${LAPACK_LIBRARIES})
endif()

add_library(funzelBlas SHARED ${SRC} ${HEADERS})
target_link_libraries(funzelBlas PUBLIC funzel PRIVATE ${FUNZEL_BLAS_LIBRARIES})

if(BLA_VENDOR STREQUAL "OpenBLAS")
	add_compile_definitions(funzelBlas PUBLIC BLAS_VENDOR_OPENBLAS)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	target_link_libraries(funzelBlas PUBLIC OpenMP::OpenMP_CXX)
endif()

if(APPLE)
	find_path(CBLAS_HEADER cblas.h)
elseif(WIN32 OR UNIX)
	# FIXME May produce errors when using Intel MKL or any other BLAS
	find_path(CBLAS_HEADER cblas.h PATH_SUFFIXES openblas)
	message("-- Looking for cblas.h: ${CBLAS_HEADER}")
endif()

target_include_directories(funzelBlas PRIVATE ${CBLAS_HEADER} ${LAPACK_INCLUDE_DIRS} ${FUNZEL_BLAS_INCLUDE_DIRS})

install(TARGETS funzelBlas
	LIBRARY DESTINATION lib
	RUNTIME DESTINATION bin
)
