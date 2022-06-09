if(NOT EXISTS ${CMAKE_BINARY_DIR}/swigwin.zip)
	message("-- Downloading SWIG")
	file(DOWNLOAD "https://downloads.sourceforge.net/project/swig/swigwin/swigwin-4.0.2/swigwin-4.0.2.zip" "${CMAKE_BINARY_DIR}/swigwin.zip" SHOW_PROGRESS)
	
	message("-- Unpacking SWIG")
	execute_process(
		COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_BINARY_DIR}/swigwin.zip
		WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
	)
endif()

set(SWIG_EXECUTABLE "${CMAKE_BINARY_DIR}/swigwin-4.0.2/swig.exe")
