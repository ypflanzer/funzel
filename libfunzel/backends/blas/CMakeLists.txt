file(GLOB SRC src/*.cpp)
file(GLOB HEADERS src/*.h*)

option(FUNZEL_NO_SIMD "Compile CPU code without SIMD optimizations." OFF)
option(FUNZEL_USE_INTERNAL_BLAS "Compile OpenBLAS ourselves instead of relying on system packages." ON)
if(FUNZEL_USE_INTERNAL_BLAS)

	add_compile_definitions(-DLAPACK_DISABLE_NAN_CHECK=1 -DLAPACK_ILP64=1 -DUSE64BITINT=1)
	set(C_LAPACK TRUE CACHE BOOL "Use C sources, Fortran may not be available." FORCE)

	option(USE_OPENMP "Use OpenMP with OpenBLAS" TRUE)
	option(USE_THREAD "Use multithreading with OpenBLAS" TRUE)
	
	if(MSVC)
		set(DYNAMIC_ARCH FALSE CACHE BOOL "Use dynamic arch for flexibility.")
	else()
		set(DYNAMIC_ARCH TRUE CACHE BOOL "Use dynamic arch for flexibility.")
	endif()

	set(NO_WARMUP TRUE CACHE BOOL "Prevent expensive warmup.")
	set(DYNAMIC_LIST "HASWELL;ZEN;SKYLAKEX" CACHE STRING "List of optimized BLAS arches.")

	if(WIN32 AND NOT MSVC)
		set(BUILD_STATIC_LIBS FALSE)
		set(FUNZEL_BLAS_LIBRARIES openblas_shared)
	else()
		set(BUILD_STATIC_LIBS TRUE)
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

	if(APPLE)
		find_path(CBLAS_HEADER cblas.h)
	elseif(WIN32 OR UNIX)
		# FIXME May produce errors when using Intel MKL or any other BLAS
		find_path(CBLAS_HEADER cblas.h PATH_SUFFIXES openblas)
		message("-- Looking for cblas.h: ${CBLAS_HEADER}")
	endif()
endif()

add_library(funzelBlas SHARED ${SRC} ${HEADERS})
target_link_libraries(funzelBlas PUBLIC funzel ${FUNZEL_BLAS_LIBRARIES})

if(FUNZEL_NO_SIMD)
	target_compile_definitions(-DFUNZEL_NO_SIMD=1)
endif()

if(BLA_VENDOR STREQUAL "OpenBLAS")
	add_compile_definitions(funzelBlas PUBLIC BLAS_VENDOR_OPENBLAS)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	target_link_libraries(funzelBlas PUBLIC OpenMP::OpenMP_CXX)
endif()

target_include_directories(funzelBlas PUBLIC ${CBLAS_HEADER} ${LAPACK_INCLUDE_DIRS} ${FUNZEL_BLAS_INCLUDE_DIRS} src)

install(TARGETS funzelBlas
	LIBRARY DESTINATION lib
	RUNTIME DESTINATION bin
)
