# The Funzel Tensor Library

Many frameworks for scientific computing, visual computing or deep learning
are heavily optimized for a small set of programming languages and
hardware platforms. Funzel tries to deliver a solution for all situations in which
none of those solutions can provide proper support because no sufficient hardware
support exists or because bindings for the programming language of choice
are missing.

# Scope

The main objective is about providing all essential building blocks
for scientific use cases, statistics, computer vision and deep learning
in a portable and easy to use way. A set of language bindings for
languages like Lua, Python, C# and Java should be delivered to provide
easy access for those who need to use one of those languages.

Hardware support is currently focused on CPU and OpenCL backends for
portability.  

| :warning: ATTENTION                                                                    |
|:---------------------------------------------------------------------------------------|
| This software is still in early development, breakage and missing features are common! |

# Development

The project uses CMake as its build system and is written mostly in C++.
Initial language bindings use the SWIG interface generator.

## Setup
```
git clone https://github.com/Sponk/funzel.git
cd funzel
git submodule update --init --recursive
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug # Debug, Release etc.
cmake --build .
```

## VCPKG
VCPKG supports the major three platforms, Windows, macOS and Linux, thus
the following set of packages can be used everywhere.

```
vcpkg install openblas opencl lua python3 gtest benchmark spdlog fltk
```
