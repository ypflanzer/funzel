
find_package(benchmark)

if(benchmark_FOUND)
	file(GLOB SRC src/*.cpp)

	if(FUNZEL_ENABLE_BLAS_BACKEND)
		file(GLOB BLAS_SRC src/backends/blas/*.cpp)
		set(BLAS_BACKEND funzelBlas)
	endif()

	if(FUNZEL_ENABLE_OPENCL_BACKEND)
		file(GLOB CL_SRC src/backends/opencl/*.cpp)
		set(CL_BACKEND funzelOpenCL)
	endif()

	if(FUNZEL_ENABLE_SYCL_BACKEND)
		file(GLOB SYCL_SRC src/backends/sycl/*.cpp)
		set(SYCL_BACKEND funzelSYCL)
	endif()

	add_executable(funzelBenchmark src/backends/Common.hpp ${SRC} ${BLAS_SRC} ${CL_SRC} ${SYCL_SRC})
	target_link_libraries(funzelBenchmark PUBLIC benchmark::benchmark funzel ${BLAS_BACKEND})
	add_dependencies(funzelBenchmark ${BLAS_BACKEND} ${CL_BACKEND} ${SYCL_BACKEND})
endif()

