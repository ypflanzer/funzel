
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

set(SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON)
set(CMAKE_STATIC_LIBRARY_PREFIX "")
set(CMAKE_SHARED_LIBRARY_PREFIX "")

include_directories(../include)

find_package(Lua)
if(Lua_FOUND)
	set_source_files_properties(LuaFunzel.i PROPERTIES CPLUSPLUS ON)
	swig_add_library(LuaFunzel LANGUAGE lua SOURCES LuaFunzel.i OUTPUT_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
	target_link_libraries(LuaFunzel PUBLIC funzel funzelBlas ${LUA_LIBRARIES})
	target_include_directories(LuaFunzel PRIVATE ${LUA_INCLUDE_DIR})
endif()

# Dotnet
find_program(CSC_EXE csc)
if(CSC_EXE)
	set_source_files_properties(CSFunzel.i PROPERTIES CPLUSPLUS ON)
	set(FUNZEL_SHARP_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/csfunzel)
	set(FUNZEL_DLL_FILE ${FUNZEL_SHARP_DIR}/FunzelSharp.dll)

	set(CMAKE_SWIG_FLAGS "-namespace" "FunzelSharp") # "-debug-tmsearch")
	swig_add_library(CSFunzel TYPE SHARED LANGUAGE csharp SOURCES CSFunzel.i OUTPUT_DIR ${FUNZEL_SHARP_DIR})
	target_link_libraries(CSFunzel PUBLIC funzel funzelBlas)

	add_custom_command(
		TARGET CSFunzel POST_BUILD
		COMMAND ${CSC_EXE} -out:${FUNZEL_DLL_FILE} -target:library -recurse:${FUNZEL_SHARP_DIR}/*.cs -recurse:${CMAKE_CURRENT_SOURCE_DIR}/*.cs -nologo
		MAIN_DEPENDENCY CSFunzel.i
		COMMENT "Compiling C#..."
	)

	configure_file("${CMAKE_CURRENT_SOURCE_DIR}/../tests/csharp/csharp.csproj.in" "${CMAKE_CURRENT_SOURCE_DIR}/../tests/csharp/csharp.csproj")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
set(CMAKE_SWIG_FLAGS "-py3" "-DPY3")
include_directories(${Python3_INCLUDE_DIRS})

if(Python3_FOUND)
	set_source_files_properties(PyFunzel.i PROPERTIES CPLUSPLUS ON)
	swig_add_library(PyFunzel LANGUAGE python SOURCES PyFunzel.i OUTPUT_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
	target_link_directories(PyFunzel PUBLIC ${Python3_LIBRARY_DIRS})
	target_link_libraries(PyFunzel PUBLIC funzel funzelBlas ${Python3_LIBRARIES} Python3::Module)
endif()

if(NOT WIN32)
	target_compile_options(PyFunzel PRIVATE -fvisibility=internal)
	target_compile_options(LuaFunzel PRIVATE -fvisibility=internal)
endif()
