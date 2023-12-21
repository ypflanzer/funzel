
option(FUNZEL_ENABLE_BLAS_BACKEND "Enables the BLAS backend." ON)
if(FUNZEL_ENABLE_BLAS_BACKEND)
	add_subdirectory(blas)
endif()

option(FUNZEL_ENABLE_OPENCL_BACKEND "Enables the OpenCL backend." ON)
if(FUNZEL_ENABLE_OPENCL_BACKEND)
	add_subdirectory(opencl)
endif()

option(FUNZEL_ENABLE_SYCL_BACKEND "Enables the SYCL backend." OFF)
if(FUNZEL_ENABLE_SYCL_BACKEND)
	add_subdirectory(sycl)
endif()

option(FUNZEL_ENABLE_HIP_BACKEND "Enables the HIP backend." OFF)
if(FUNZEL_ENABLE_HIP_BACKEND)
	add_subdirectory(hip)
endif()
