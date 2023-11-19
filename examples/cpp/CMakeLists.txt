
file(GLOB SOURCES "*.cpp")
foreach(SPATH ${SOURCES})
	get_filename_component(S "${SPATH}" NAME_WE)

	message("-- Example: ${S}")

	add_executable(${S}_example "${SPATH}")
	target_link_libraries(${S}_example funzel)
	add_dependencies(${S}_example funzelBlas)
endforeach()
