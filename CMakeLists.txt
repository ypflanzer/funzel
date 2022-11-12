cmake_minimum_required(VERSION 3.22)

option(USE_VCPKG "Use VCPKG instead of system packages for dependencies." OFF)
if(WIN32 OR USE_VCPKG)

	if(WIN32)
		set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "The VCPKG target triplet to use.")
	elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
		set(VCPKG_TARGET_TRIPLET "x64-linux-libcpp" CACHE STRING "The VCPKG target triplet to use.")
	endif()

	set(CMAKE_TOOLCHAIN_FILE
			"${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
			CACHE PATH "The VCPKG installation to use.")
endif()

project(
	Funzel
	VERSION 0.1
	DESCRIPTION "A small, GPU enabled tensor library for C++."	
)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

## Set output directories so all artifacts are easily found
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(USE_LIBCPP "Use LLVM libc++ instead of libstdc++." OFF)
if (USE_LIBCPP AND CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT ANDROID)
	add_compile_options(-stdlib=libc++)
	add_link_options(-stdlib=libc++)
endif()

## Set some optimization flags for MSVC
if(MSVC)
	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /O2 /Ob2 /Oi /Ot")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
	option(ENABLE_AVX2 "Use AVX2 instruction set." ON)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2") # "-march=native")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_SANITIZERS "Enable Clang sanitizers." OFF)

if(WIN32)
	include(DownloadSwig)
endif()

add_subdirectory(libfunzel)
add_subdirectory (docs)
