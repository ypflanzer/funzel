file(GLOB SRC src/*.cpp)

if(APPLE)
	set(CMAKE_PREFIX_PATH "/usr/local/opt/openblas")
endif()

set(BLA_VENDOR "OpenBLAS" CACHE STRING "The BLAS implementation to be used.")
find_package(BLAS REQUIRED)

add_library(funzelBlas SHARED ${SRC})
target_link_libraries(funzelBlas PUBLIC funzel PRIVATE BLAS::BLAS)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	target_link_libraries(funzelBlas PUBLIC OpenMP::OpenMP_CXX)
endif()

if(APPLE)
	find_path(CBLAS_HEADER cblas.h)
	target_include_directories(funzelBlas PRIVATE ${CBLAS_HEADER})
endif()
