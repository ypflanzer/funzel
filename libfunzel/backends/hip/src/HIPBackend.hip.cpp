#include <funzel/Funzel.hpp>
#include "HIPTensor.hip.hpp"

#include <iostream>

using namespace funzel;
using namespace funzel::hip;

FUNZEL_REGISTER_BACKEND("HIP", HIPTensor)
