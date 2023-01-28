
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

set(SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON)

set(CMAKE_STATIC_LIBRARY_PREFIX "")
set(CMAKE_SHARED_LIBRARY_PREFIX "")

include_directories(../include)

find_package(Lua)
if(LUA_FOUND AND NOT NO_LUA)
	set_source_files_properties(LuaFunzel.i PROPERTIES CPLUSPLUS ON)
	swig_add_library(LuaFunzel LANGUAGE lua SOURCES LuaFunzel.i OUTPUT_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
	target_include_directories(LuaFunzel PRIVATE ${LUA_INCLUDE_DIR})

	if(NOT WIN32)
		target_link_libraries(LuaFunzel PUBLIC funzel funzelBlas)
		target_link_options(LuaFunzel PUBLIC ${LUA_LIBRARIES}) # options, or else CMake will make a wrong link to libm.
	else()
		target_link_libraries(LuaFunzel PUBLIC funzel funzelBlas ${LUA_LIBRARIES})
	endif()

	install(TARGETS LuaFunzel DESTINATION lib)

	if(NOT WIN32)
		target_compile_options(LuaFunzel PRIVATE -fvisibility=internal)
	endif()
else()
	message("-- Could not find Lua")
endif()

# Dotnet
if(NOT NO_CSHARP)
	if(WIN32)
		enable_language(CSharp)
		set(CSC_EXE ${CMAKE_CSharp_COMPILER})
		# find_program(CSC_EXE csc HINTS "C:/WINDOWS/Microsoft.NET/Framework64/")
	else()
		find_program(CSC_EXE csc)
	endif()

	if(CSC_EXE)
		set_source_files_properties(CSFunzel.i PROPERTIES CPLUSPLUS ON)
		set(FUNZEL_SHARP_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/csfunzel)
		set(FUNZEL_DLL_FILE ${FUNZEL_SHARP_DIR}/../FunzelSharp.dll)

		#set(CMAKE_SWIG_FLAGS "-namespace" "funzel") # "-debug-tmsearch")
		swig_add_library(CSFunzel TYPE SHARED LANGUAGE csharp SOURCES CSFunzel.i OUTPUT_DIR ${FUNZEL_SHARP_DIR})
		target_link_libraries(CSFunzel PUBLIC funzel funzelBlas)

		add_custom_command(
			TARGET CSFunzel POST_BUILD
			COMMAND ${CSC_EXE} -out:${FUNZEL_DLL_FILE} -o+ -target:library -recurse:${FUNZEL_SHARP_DIR}/*.cs -recurse:${CMAKE_CURRENT_SOURCE_DIR}/*.cs -nologo
			MAIN_DEPENDENCY CSFunzel.i
			COMMENT "Compiling C#..."
		)

		configure_file("${CMAKE_CURRENT_SOURCE_DIR}/../tests/csharp/csharp.csproj.in" "${CMAKE_CURRENT_SOURCE_DIR}/../tests/csharp/csharp.csproj")
		
		install(TARGETS CSFunzel DESTINATION lib)
		install(FILES ${FUNZEL_DLL_FILE} DESTINATION lib)

		if(NOT WIN32)
			target_compile_options(CSFunzel PRIVATE -fvisibility=internal)
		endif()
	endif()
endif()

if(NOT NO_PYTHON)
	find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
	set(CMAKE_SWIG_FLAGS "-py3" "-DPY3")
	include_directories(${Python3_INCLUDE_DIRS})

	if(Python3_FOUND)
		set_source_files_properties(PyFunzel.i PROPERTIES CPLUSPLUS ON)
		swig_add_library(PyFunzel LANGUAGE python SOURCES PyFunzel.i OUTPUT_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
		target_link_directories(PyFunzel PUBLIC ${Python3_LIBRARY_DIRS})
		target_link_libraries(PyFunzel PUBLIC funzel funzelBlas ${Python3_LIBRARIES} Python3::Python)

		install(TARGETS PyFunzel DESTINATION lib)
		install(FILES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/PyFunzel.py DESTINATION lib)

		if(NOT WIN32)
			target_compile_options(PyFunzel PRIVATE -fvisibility=internal)
		endif()
	endif()
endif()

find_package(JNI)
find_package(Java)

if(NOT NO_JAVA AND JNI_FOUND AND Java_FOUND)
	set(CMAKE_SWIG_FLAGS "-package" "JFunzel")
	set(FUNZEL_JAVA_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/jfunzel/JFunzel)

	set_source_files_properties(JavaFunzel.i PROPERTIES CPLUSPLUS ON)
	swig_add_library(JFunzel LANGUAGE java SOURCES JavaFunzel.i OUTPUT_DIR ${FUNZEL_JAVA_DIR})
	target_link_libraries(JFunzel PUBLIC funzel funzelBlas JNI::JNI)

	add_custom_command(
		TARGET JFunzel POST_BUILD
		COMMAND ${Java_JAVAC_EXECUTABLE} ${FUNZEL_JAVA_DIR}/*.java
		COMMAND ${Java_JAR_EXECUTABLE} cf ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/JFunzel.jar ${FUNZEL_JAVA_DIR}/*.class
		MAIN_DEPENDENCY JavaFunzel.i
		COMMENT "Compiling Java..."
	)

	install(TARGETS JFunzel DESTINATION lib)
	install(FILES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/JFunzel.jar DESTINATION lib)
endif()
