file(GLOB SRC src/*.cpp)

find_package(BLAS REQUIRED)

add_library(funzelBlas SHARED ${SRC})
target_link_libraries(funzelBlas PUBLIC funzel PRIVATE BLAS::BLAS)


find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	target_link_libraries(funzelBlas PUBLIC OpenMP::OpenMP_CXX)
endif()
