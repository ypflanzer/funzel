file(GLOB_RECURSE SRC src/*.cpp)
file(GLOB_RECURSE HEADERS include/funzel/*.h*)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-sign-compare")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-sign-compare")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT LTO_ENABLED OUTPUT error)

if(LTO_ENABLED AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
	message(STATUS "IPO / LTO enabled")
	set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
	message(STATUS "IPO / LTO not supported")
endif()

## Build MathGL
#include(ExternalProject)
#ExternalProject_Add(MathGL
#	SVN_REPOSITORY https://svn.code.sf.net/p/mathgl/code/
#	SVN_REVISION ""
#	SOURCE_SUBDIR mathgl-2x
#	CMAKE_ARGS -Denable-lgpl=TRUE
#	INSTALL_COMMAND ""
#)

#include(FetchContent)
#set(FETCHCONTENT_QUIET FALSE)
#FetchContent_Declare(
#	MathGL
#	SVN_REPOSITORY https://svn.code.sf.net/p/mathgl/code/
#	SVN_REVISION ""
#)
#FetchContent_MakeAvailable(MathGL)

find_package(spdlog REQUIRED)
find_package(FLTK REQUIRED)

add_library(funzel SHARED ${SRC} ${HEADERS})
message("-- FLTK: ${FLTK_INCLUDE_DIR}")
target_include_directories(funzel PUBLIC include PRIVATE ${FLTK_INCLUDE_DIR})
target_link_libraries(funzel PUBLIC spdlog::spdlog PRIVATE ${FLTK_LIBRARIES})

target_compile_definitions(funzel PRIVATE FUNZEL_EXPORT=1)

if(NOT WIN32)
	target_compile_options(funzel PRIVATE -fvisibility=internal)
endif()

## MathGL
#set(enable-lgpl TRUE CACHE BOOL "" FORCE)
#set(enable-mgl2 TRUE CACHE BOOL "" FORCE)
#set(enable-fltk TRUE CACHE BOOL "" FORCE)
#set(enable-qt TRUE CACHE BOOL "" FORCE)
#set(enable-glut TRUE CACHE BOOL "" FORCE)

#add_subdirectory(${mathgl_SOURCE_DIR}/mathgl-2x ${mathgl_BINARY_DIR} EXCLUDE_FROM_ALL)
#target_link_libraries(funzel PUBLIC mgl mgl-fltk mgl-qt5 mgl-glut)
#target_include_directories(funzel PUBLIC ${mathgl_SOURCE_DIR}/mathgl-2x/include ${mathgl_BINARY_DIR}/include)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	target_link_libraries(funzel PUBLIC OpenMP::OpenMP_CXX)
endif()

if(ENABLE_SANITIZERS)
	target_compile_options(funzel PUBLIC -fsanitize=address)
	target_link_options(funzel PUBLIC -fsanitize=address)
endif()

add_subdirectory(swig)
add_subdirectory(backends)
add_subdirectory(tests)
add_subdirectory(benchmarks)

install(TARGETS funzel DESTINATION lib)
