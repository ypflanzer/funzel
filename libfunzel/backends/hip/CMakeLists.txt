# https://rocmdocs.amd.com/en/latest/Installation_Guide/Using-CMake-with-AMD-ROCm.html
list(APPEND CMAKE_MODULE_PATH "/opt/rocm/hip/cmake" "/opt/rocm/lib/cmake")
list(APPEND CMAKE_PREFIX_PATH "/opt/rocm/hip/cmake" "/opt/rocm/lib/cmake")

# https://github.com/ROCm-Developer-Tools/HIP/issues/3141
# It seems to need this in PATH
set(ENV{PATH} "$ENV{PATH}:/opt/rocm")

set(CMAKE_HIP_ARCHITECTURES "gfx900" CACHE STRING "GPU targets to compile for")
#set(CMAKE_HIP_COMPILER_ROCM_ROOT "/opt/rocm")
set(HIP_ROOT_DIR "/opt/rocm")
set(HIP_PATH "/opt/rocm")

if(FALSE)
find_package(HIP)

if(HIP_FOUND)
	find_package(hipblas REQUIRED)
	message("-- Found HIP: ${HIP_VERSION}")

	file(GLOB SRC src/*.cpp)
	set_source_files_properties(${SRC} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
	
	hip_add_library(funzelHIP SHARED ${SRC} HIPCC_OPTIONS -std=c++17)
	
	target_link_libraries(funzelHIP PUBLIC funzel roc::hipblas hip::device)
	target_include_directories(funzelHIP PUBLIC src)
	target_compile_options(funzelHIP PRIVATE -std=c++17)

else()
	message("-- Could not find HIP")
endif()
endif()

find_package(HIP)

include(CheckLanguage)
check_language(HIP)
enable_language(HIP)

if(CMAKE_HIP_COMPILER)
	enable_language(HIP)
	message("-- Using HIP")
else()
	message(STATUS "No HIP support.")
	return()
endif()

file(GLOB SRC src/*.cpp)
file(GLOB HEADERS src/*.hpp)

find_package(HIP REQUIRED)
find_package(hipblas REQUIRED)

set_source_files_properties(${SRC} PROPERTIES LANGUAGE HIP)

add_library(funzelHIP SHARED ${SRC} ${HEADERS})
target_link_libraries(funzelHIP PUBLIC funzel roc::hipblas hip::device)
target_include_directories(funzelHIP PUBLIC src)
#target_link_directories(funzelHIP PUBLIC "/usr/lib/clang/13.0.1/lib/linux/")

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	target_link_libraries(funzelHIP PUBLIC OpenMP::OpenMP_CXX)
endif()
