
add_executable(funzel-ls main.cpp)
target_link_libraries(funzel-ls PUBLIC funzel funzelBlas funzelOpenCL)
