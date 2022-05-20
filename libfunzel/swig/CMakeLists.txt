
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

set(SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON)

include_directories(../include)

set_source_files_properties(LuaFunzel.i PROPERTIES CPLUSPLUS ON)
swig_add_library(LuaFunzel LANGUAGE lua SOURCES LuaFunzel.i)
target_link_libraries(LuaFunzel PUBLIC funzel funzelBlas)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
list(APPEND CMAKE_SWIG_FLAGS "-py3" "-DPY3")
include_directories(${Python3_INCLUDE_DIRS})

if(Python3_FOUND)
	set_source_files_properties(PyFunzel.i PROPERTIES CPLUSPLUS ON)
	swig_add_library(PyFunzel LANGUAGE python SOURCES PyFunzel.i)
	target_link_directories(PyFunzel PUBLIC ${Python3_LIBRARY_DIRS})
	target_link_libraries(PyFunzel PUBLIC funzel funzelBlas ${Python3_LIBRARIES})
endif()
