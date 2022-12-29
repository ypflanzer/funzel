file(GLOB SRC src/*.cpp)
file(GLOB HEADERS src/*.h*)

if(APPLE)
	set(CMAKE_PREFIX_PATH "/usr/local/opt/openblas")
endif()

set(BLA_VENDOR "OpenBLAS" CACHE STRING "The BLAS implementation to be used.")
find_package(BLAS REQUIRED)

add_library(funzelBlas SHARED ${SRC} ${HEADERS})
target_link_libraries(funzelBlas PUBLIC funzel PRIVATE BLAS::BLAS)

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

target_include_directories(funzelBlas PRIVATE ${CBLAS_HEADER})

install(TARGETS funzelBlas
	LIBRARY DESTINATION lib
	RUNTIME DESTINATION bin
)
