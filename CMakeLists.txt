cmake_minimum_required(VERSION 3.22)

if(WIN32)
	set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "The VCPKG target triplet to use.")
	set(CMAKE_TOOLCHAIN_FILE
			"${CMAKE_SOURCE_DIR}/../vcpkg/scripts/buildsystems/vcpkg.cmake"
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

## Set some optimization flags for MSVC
if(MSVC)
	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/O2 /Ob2 /Oi /Ot")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_SANITIZERS "Enable Clang sanitizers." OFF)

if(WIN32)
	include(DownloadSwig)
endif()

add_subdirectory(libfunzel)
add_subdirectory (docs)
