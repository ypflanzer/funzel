
find_package(GTest CONFIG)

if(GTest_FOUND)
	file(GLOB SRC libfunzel/*.cpp)
	set(BACKENDS "\n")
	if(FUNZEL_ENABLE_BLAS_BACKEND)
		file(GLOB BLAS_SRC backends/blas/*.cpp)
		set(BLAS_BACKEND funzelBlas)
		STRING(APPEND BACKENDS "\tfunzel::backend::LoadBackend(\"Blas\");\n")
	endif()

	if(FUNZEL_ENABLE_OPENCL_BACKEND)
		file(GLOB CL_SRC backends/opencl/*.cpp)
		set(CL_BACKEND funzelOpenCL)
		STRING(APPEND BACKENDS "\tfunzel::backend::LoadBackend(\"OpenCL\");\n")
	endif()

	if(FUNZEL_ENABLE_SYCL_BACKEND)
		file(GLOB SYCL_SRC backends/sycl/*.cpp)
		set(SYCL_BACKEND funzelSYCL)

		STRING(APPEND BACKENDS "\tfunzel::backend::LoadBackend(\"SYCL\");\n")
	endif()

	if(FUNZEL_ENABLE_HIP_BACKEND)
		file(GLOB HIP_SRC backends/hip/*.cpp)
		set(HIP_BACKEND funzelHIP)

		STRING(APPEND BACKENDS "\tfunzel::backend::LoadBackend(\"HIP\");\n")
	endif()

	configure_file(LoadBackends.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/LoadBackends.hpp @ONLY)

	add_executable(funzelTest Common.hpp ${SRC} ${BLAS_SRC} ${CL_SRC} ${SYCL_SRC} ${HIP_SRC})
	target_link_libraries(funzelTest PUBLIC GTest::gtest funzel ${BLAS_BACKEND} ${CL_BACKEND} ${SYCL_BACKEND} ${HIP_BACKEND})
	target_include_directories(funzelTest PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
	add_dependencies(funzelTest ${BLAS_BACKEND} ${CL_BACKEND} ${SYCL_BACKEND} ${HIP_BACKEND})

	# Copy test data to binary dir
	add_custom_command(
		TARGET funzelTest
		POST_BUILD
		COMMAND ${CMAKE_COMMAND}
		ARGS -E copy_directory_if_different ${CMAKE_CURRENT_SOURCE_DIR}/data $<TARGET_FILE_DIR:funzelTest>/data
		COMMENT "Copying test data..."
	)

	if(NOT WIN32)
		add_custom_target(kcov
			COMMAND rm -rf ${CMAKE_BINARY_DIR}/kcov && kcov --include-pattern=funzel --exclude-pattern=CLBlast,tests,vcpkg,stb_image,openblas ${CMAKE_BINARY_DIR}/kcov $<TARGET_FILE:funzelTest>
			
			WORKING_DIRECTORY $<TARGET_FILE_DIR:funzelTest>
			DEPENDS funzelTest
		)

		find_program(PERF perf REQUIRED)
		if(PERF_FOUND)
			add_custom_target(perf
				COMMAND ${PERF} record -F 99 -g $<TARGET_FILE:funzelTest>
				WORKING_DIRECTORY $<TARGET_FILE_DIR:funzelTest>
				DEPENDS funzelTest
			)
		endif()

		if(ENABLE_PROFILING)
			find_program(GPROF gprof REQUIRED)
			add_custom_target(gprof
				COMMAND $<TARGET_FILE:funzelTest> || exit 0
				COMMAND ${GPROF} --demangle $<TARGET_FILE:funzelTest> > ${CMAKE_BINARY_DIR}/gprof_report.txt
				WORKING_DIRECTORY $<TARGET_FILE_DIR:funzelTest>
				DEPENDS funzelTest
			)
		else()
			set(GPROF "")
		endif()

		if(ENABLE_GCOV)
			find_program(LCOV lcov REQUIRED)
			find_program(GENHTML genhtml REQUIRED)

			add_custom_target(lcov
				COMMAND $<TARGET_FILE:funzelTest> || exit 0
				COMMAND ${LCOV}
					--include "libfunzel"
					--directory ${CMAKE_BINARY_DIR}
					--base-directory ${CMAKE_SOURCE_DIR}
					--capture --parallel 8
					--branch-coverage
					--keep-going
					--quiet
					--output-file ${CMAKE_BINARY_DIR}/coverage.info
					--ignore-errors gcov,mismatch
				COMMAND ${LCOV} --remove ${CMAKE_BINARY_DIR}/coverage.info -o ${CMAKE_BINARY_DIR}/coverage.info "*CLBlast*" "*tests*" "*stb_*" --ignore-errors unused
				COMMAND ${GENHTML} --demangle-cpp -o ${CMAKE_BINARY_DIR}/coverage ${CMAKE_BINARY_DIR}/coverage.info

				WORKING_DIRECTORY $<TARGET_FILE_DIR:funzelTest>
				DEPENDS funzelTest
			)
		endif()
	endif()
endif()
