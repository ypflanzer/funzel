# https://rocmdocs.amd.com/en/latest/Installation_Guide/Using-CMake-with-AMD-ROCm.html
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
set(CMAKE_HIP_ARCHITECTURES "gfx900" CACHE STRING "GPU targets to compile for")

include(CheckLanguage)
check_language(HIP)
if(CMAKE_HIP_COMPILER)
	enable_language(HIP)
	message("-- Using HIP")
else()
	message(STATUS "No HIP support.")
	return()
endif()

file(GLOB SRC src/*.cpp)

find_package(hip REQUIRED)
find_package(hipblas REQUIRED)

set_source_files_properties(${SRC} PROPERTIES LANGUAGE HIP)

add_library(funzelHIP SHARED ${SRC})
target_link_libraries(funzelHIP PUBLIC funzel roc::hipblas hip::device)
target_include_directories(funzelHIP PUBLIC src)
#target_link_directories(funzelHIP PUBLIC "/usr/lib/clang/13.0.1/lib/linux/")

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	target_link_libraries(funzelHIP PUBLIC OpenMP::OpenMP_CXX)
endif()
