#include "gtest/gtest.h"
#include <funzel/Funzel.hpp>

int main(int argc, char **argv)
{
	funzel::backend::LoadBackend("Blas");
	funzel::backend::LoadBackend("OpenCL");

	::testing::InitGoogleTest(&argc, argv);
	int ret = RUN_ALL_TESTS();
	return ret;
}
