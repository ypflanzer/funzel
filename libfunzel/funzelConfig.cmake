include(CMakeFindDependencyMacro)

## Insert all dependencies here.
find_dependency(spdlog CONFIG REQUIRED)
find_dependency(FLTK REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/funzel-targets.cmake")
