
add_subdirectory(funzel-ls)

option(NO_EXAMPLES "Disable building all examples." FALSE)
if(NOT NO_EXAMPLES)
	add_subdirectory(cpp)
endif()
