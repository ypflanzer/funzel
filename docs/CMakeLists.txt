
## Doxygen part
find_package(Doxygen REQUIRED)
if(NOT Doxygen_FOUND)
	message(STATUS "Doxygen was not found, documentation is disabled.")
	return()
endif()

## Find all headers to depend on
get_target_property(PUBLIC_HEADER_DIR funzel INTERFACE_INCLUDE_DIRECTORIES)
file(GLOB_RECURSE PUBLIC_HEADERS ${PUBLIC_HEADER_DIR}/*.h*)

## Set all doxygen parameters for generating XML
set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR}/libfunzel/include/funzel)

set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIRECTORY}/xml/index.xml)

set(DOXYGEN_GENERATE_HTML YES)
set(DOXYGEN_GENERATE_XML YES)
set(DOXYGEN_USE_MATHJAX YES)

file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIRECTORY}/xml/)
file(TOUCH ${DOXYGEN_INDEX_FILE})

doxygen_add_docs(
	doxygen
	${DOXYGEN_INPUT_DIR}
	COMMENT "Generating Doxygen documentation."
)

## Sphinx part
find_package(Sphinx)
if(NOT Sphinx_FOUND)
	message(STATUS "Sphinx was not found.")
	return()
endif()

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx)
set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)
set(SPHINX_AUTODOC ${CMAKE_CURRENT_SOURCE_DIR}/sphinx_autodoc)

macro(add_breathe_apidoc PROJECT XML_DIR OUT_DIR)
	file(MAKE_DIRECTORY ${OUT_DIR})
	add_custom_command(OUTPUT ${OUT_DIR}/classlist.rst
		COMMAND # TODO Proper find!
			breathe-apidoc -f -o ${OUT_DIR} -m -p ${PROJECT} ${XML_DIR}
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		MAIN_DEPENDENCY
			${DOXYGEN_INDEX_FILE}
		COMMENT "Generating documentation with breathe-apidoc")
endmacro()

macro(add_sphinx_apidoc PYSRC_DIR OUT_DIR)
	file(MAKE_DIRECTORY ${OUT_DIR})
	add_custom_command(OUTPUT ${OUT_DIR}/modules.rst
		COMMAND # TODO Proper find!
			sphinx-apidoc -f -o ${OUT_DIR} ${PYSRC_DIR}
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		MAIN_DEPENDENCY
			${DOXYGEN_INDEX_FILE}
		COMMENT "Generating Python documentation with sphinx-apidoc")
endmacro()

add_sphinx_apidoc(${CMAKE_CURRENT_BINARY_DIR}/../libfunzel/swig/ ${SPHINX_AUTODOC}_py)

add_breathe_apidoc(FunzelCpp ${DOXYGEN_OUTPUT_DIRECTORY}/xml ${SPHINX_AUTODOC}_cpp)
#add_breathe_apidoc(FunzelLua ${DOXYGEN_OUTPUT_DIRECTORY}/lua_xml ${SPHINX_AUTODOC}_lua)

add_custom_command(OUTPUT ${SPHINX_INDEX_FILE}
					COMMAND
						${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/../libfunzel/swig LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/../libfunzel
						${SPHINX_EXECUTABLE} -b html
						-Dbreathe_projects.FunzelCpp=${DOXYGEN_OUTPUT_DIRECTORY}/xml
						#-Dbreathe_projects.FunzelLua=${DOXYGEN_OUTPUT_DIRECTORY}/lua_xml

					${SPHINX_SOURCE} ${SPHINX_BUILD}
					WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
					DEPENDS
						${DOXYGEN_INDEX_FILE}
						${SPHINX_AUTODOC}_cpp/classlist.rst
						#${SPHINX_AUTODOC}_lua/classlist.rst
						#${SPHINX_AUTODOC}_py/modules.rst
						${CMAKE_CURRENT_SOURCE_DIR}/index.rst
						${CMAKE_CURRENT_SOURCE_DIR}/../README.md

					MAIN_DEPENDENCY
						${SPHINX_SOURCE}/conf.py
					COMMENT "Generating documentation with Sphinx")

add_custom_target(sphinx DEPENDS ${SPHINX_INDEX_FILE})
add_dependencies(sphinx doxygen)
