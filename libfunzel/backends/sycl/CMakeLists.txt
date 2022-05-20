
set(HIPSYCL_TARGETS "omp;hip:gfx900" CACHE STRING "The target platforms for the SYCL backend.")

find_package(hipSYCL CONFIG REQUIRED)
message("-- Using hipSYCL")

set(BLAS_ENABLE_TESTING FALSE CACHE BOOL "" FORCE)
set(BLAS_ENABLE_BENCHMARK FALSE CACHE BOOL "" FORCE)
set(BLAS_VERIFY_BENCHMARK FALSE CACHE BOOL "" FORCE)
set(BLAS_BUILD_SAMPLES FALSE CACHE BOOL "" FORCE)

#add_subdirectory(sycl-blas EXCLUDE_FROM_ALL)

file(GLOB SRC src/*.cpp)

add_library(funzelSYCL SHARED ${SRC})
target_link_libraries(funzelSYCL PUBLIC funzel)
target_include_directories(funzelSYCL PUBLIC src)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	target_link_libraries(funzelSYCL PUBLIC OpenMP::OpenMP_CXX)
endif()

add_sycl_to_target(TARGET funzelSYCL SOURCES ${SRC})
