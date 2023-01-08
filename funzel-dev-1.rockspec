package = "funzel"
version = "dev-1"
source = {
   url = "git+ssh://git@github.com/Sponk/funzel.git"
}
description = {
   summary = "Many frameworks for scientific computing, visual computing or deep learning are heavily optimized for a small set of programming languages and hardware platforms.",
   detailed = [[
Many frameworks for scientific computing, visual computing or deep learning
are heavily optimized for a small set of programming languages and
hardware platforms. Funzel tries to deliver a solution for all situations in which
none of those solutions can provide proper support because no sufficient hardware
support exists or because bindings for the programming language of choice
are missing.]],
   homepage = "https://www.github.com/sponk/funzel",
   license = "LGPLv3"
}
build = {
   type = "cmake",
   variables = {
      ["CMAKE_INSTALL_PREFIX"] = "$(PREFIX)",
      ["NO_PYTHON"] = "TRUE",
      ["NO_CSHARP"] = "TRUE",
      ["NO_EXAMPLES"] = "TRUE",
      ["NO_TESTS"] = "TRUE",
      ["BUILD_SHARED_LIBS"] = "FALSE" -- To prevent issues when installing symlinks.
   }
}
