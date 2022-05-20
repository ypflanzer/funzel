#include <funzel/Funzel.hpp>
#include "SYCLBackend.hpp"
#include "SYCLTensor.hpp"

#include <iostream>

using namespace funzel;
using namespace funzel::sycl;

FUNZEL_REGISTER_BACKEND("SYCL", SYCLTensor)
