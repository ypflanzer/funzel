
find_package(GTest)

if(GTest_FOUND)
	file(GLOB SRC libfunzel/*.cpp)

	if(ENABLE_BLAS_BACKEND)
		file(GLOB BLAS_SRC backends/blas/*.cpp)
		set(BLAS_BACKEND funzelBlas)
	endif()

	if(ENABLE_OPENCL_BACKEND)
		file(GLOB CL_SRC backends/opencl/*.cpp)
		set(CL_BACKEND funzelOpenCL)
	endif()

	if(ENABLE_SYCL_BACKEND)
		file(GLOB SYCL_SRC backends/sycl/*.cpp)
		set(SYCL_BACKEND funzelSYCL)
	endif()

	if(ENABLE_HIP_BACKEND)
		file(GLOB HIP_SRC backends/hip/*.cpp)
		set(HIP_BACKEND funzelHIP)
	endif()

	add_executable(funzelTest Common.hpp ${SRC} ${BLAS_SRC} ${CL_SRC} ${SYCL_SRC} ${HIP_SRC})
	target_link_libraries(funzelTest PUBLIC GTest::gtest funzel ${BLAS_BACKEND} ${CL_BACKEND} ${SYCL_BACKEND} ${HIP_BACKEND})
endif()
