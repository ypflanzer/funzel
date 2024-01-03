file(GLOB SRC src/*.cpp)
file(GLOB INLINE_SRC src/*.inl)
file(GLOB HEADERS src/*.h*)

option(FUNZEL_NO_SIMD "Compile CPU code without SIMD optimizations." OFF)
option(FUNZEL_USE_INTERNAL_BLAS "Compile OpenBLAS ourselves instead of relying on system packages." OFF)
option(FUNZEL_USE_SHARED_BLAS "Link to BLAS as a shared library. May resort to dynamic linking of no static lib is found." OFF)

## Building OpenBLAS needs to adda post-build step so the target
## needs to be created first.
add_library(funzelBlas SHARED ${SRC} ${HEADERS} ${INLINE_SRC})

if(FUNZEL_USE_INTERNAL_BLAS)
	include(build_openblas.cmake)
else()
	if(APPLE)
		set(CMAKE_PREFIX_PATH "/usr/local/opt/openblas")
	endif()

	set(BLA_SIZEOF_INTEGER "ANY" CACHE STRING "The size of an integer. Selects 32bit vs 64bit.")
	set(BLA_VENDOR "OpenBLAS" CACHE STRING "The BLAS implementation to be used.")
	set(BLA_STATIC ${FUNZEL_USE_SHARED_BLAS})

	find_package(BLAS REQUIRED)
	find_package(LAPACK REQUIRED)

	set(FUNZEL_BLAS_LIBRARIES BLAS::BLAS ${LAPACK_LIBRARIES})

	if(BLA_VENDOR STREQUAL "OpenBLAS")
		if(APPLE)
			find_path(CBLAS_HEADER cblas.h)
		elseif(WIN32 OR UNIX)
			# FIXME May produce errors when using Intel MKL or any other BLAS
			find_path(CBLAS_HEADER cblas.h PATH_SUFFIXES openblas)
			message("-- Looking for cblas.h: ${CBLAS_HEADER}")
		endif()
	endif()
endif()

target_link_libraries(funzelBlas PUBLIC funzel ${FUNZEL_BLAS_LIBRARIES})
target_compile_definitions(funzelBlas PUBLIC ${FUNZEL_BLAS_DEFINITIONS})

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

find_package(GTest CONFIG)
if(GTest_FOUND)
	file(GLOB TEST_SRC tests/*.cpp)
	add_executable(blasBackendTest ${TEST_SRC})
	target_link_libraries(blasBackendTest PUBLIC funzelBlas GTest::gtest_main)
endif()
