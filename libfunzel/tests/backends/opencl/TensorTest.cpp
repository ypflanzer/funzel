#include <gtest/gtest.h>
#include <funzel/Tensor.hpp>
#include <funzel/OpenCLBackend.hpp>

using namespace funzel;

TEST(OpenCLTensor, Init)
{
	// funzel::cl::OpenCLBackend::initialize();

	auto ones = Tensor::ones({3, 3, 3}, funzel::FLOAT32, "OCL:0");
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

TEST(OpenCLTensor, Abs)
{
	//funzel::cl::OpenCLBackend::initialize();

	auto v = Tensor::empty({3, 3, 3});

	for(size_t p = 0; p < 3; p++)
		for(size_t q = 0; q < 3; q++)
			for(size_t r = 0; r < 3; r++)
			{
				v[{p, q, r}] = -1;
			}

	auto vcl = v.to("OCL:0");

	auto vabs = vcl[0].abs().cpu();
	std::cout << vabs << std::endl;

	//std::cout << "Abs!" << std::endl;
	//auto bigv = Tensor::empty({3000, 30000}, FLOAT32, "OpenCL:0");
	//bigv.abs_();
}

#include <funzel/Image.hpp>
#include <funzel/Plot.hpp>

TEST(OpenCLTensor, Conv2d)
{
	auto img = image::load("test.jpg").to("OCL:0");

	img->conv2d(img, tgt, kernel, {1, 1}, {0, 0}, {1, 1});

	Plot plt;
	plt.image(img.cpu());
	plt.show();
}

#define CommonTest CLTensorTest
#define TestDevice "OCL:0"
#include "../../Common.hpp"

