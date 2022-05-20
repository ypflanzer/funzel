
option(ENABLE_BLAS_BACKEND "Enables the BLAS backend." ON)
if(ENABLE_BLAS_BACKEND)
	add_subdirectory(blas)
endif()

option(ENABLE_OPENCL_BACKEND "Enables the OpenCL backend." ON)
if(ENABLE_OPENCL_BACKEND)
	add_subdirectory(opencl)
endif()

option(ENABLE_SYCL_BACKEND "Enables the SYCL backend." OFF)
if(ENABLE_SYCL_BACKEND)
	add_subdirectory(sycl)
endif()

option(ENABLE_HIP_BACKEND "Enables the HIP backend." OFF)
if(ENABLE_HIP_BACKEND)
	add_subdirectory(hip)
endif()
