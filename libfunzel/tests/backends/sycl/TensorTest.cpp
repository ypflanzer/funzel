#include <gtest/gtest.h>
#include <funzel/Tensor.hpp>

using namespace funzel;

#define CommonTest SYCLTensorTest
#define TestDevice "SYCL:0"
#include "../../Common.hpp"
