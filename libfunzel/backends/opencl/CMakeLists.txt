file(GLOB SRC src/*.cpp)

# Policy for new project command
set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)

# IPO policy for enabling LTO
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

find_package(OpenCL REQUIRED)
add_subdirectory(CLBlast)

add_library(funzelOpenCL SHARED ${SRC})
target_link_libraries(funzelOpenCL PUBLIC funzel OpenCL::OpenCL PRIVATE clblast)
target_include_directories(funzelOpenCL PUBLIC src)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	target_link_libraries(funzelOpenCL PUBLIC OpenMP::OpenMP_CXX)
endif()
