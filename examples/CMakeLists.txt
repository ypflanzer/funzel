
add_subdirectory(funzel-ls)

option(FUNZEL_NO_EXAMPLES "Disable building all examples." FALSE)
if(NOT FUNZEL_NO_EXAMPLES)
	add_subdirectory(cpp)
endif()
