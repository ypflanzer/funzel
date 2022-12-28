#include <gtest/gtest.h>
#include <funzel/Tensor.hpp>
#include <funzel/OpenCLBackend.hpp>

using namespace funzel;

#define TestDevice "OCL:0"
#define CommonTest CLTensorTest

TEST(CommonTest, Init)
{
	// funzel::cl::OpenCLBackend::initialize();

	auto ones = Tensor::ones({3, 3, 3}, funzel::DFLOAT32, TestDevice);
	EXPECT_EQ(ones.strides, Shape({36, 12, 4}));
	EXPECT_EQ(ones.shape, Shape({3, 3, 3}));

	std::cout << ones.cpu() << std::endl;

	//EXPECT_THROW((ones[{0, 0, 0}].item<float>()), std::runtime_error);

	auto onesCpu = ones.cpu();
	EXPECT_EQ(onesCpu.strides, ones.strides);
	EXPECT_EQ(onesCpu.shape, ones.shape);

	auto oneOnes = ones[{0, 0, 0}];
	auto oneOnesCpu = oneOnes.cpu();

	EXPECT_EQ((oneOnesCpu.item<float>()), 1);
}

#include "../../Common.hpp"

