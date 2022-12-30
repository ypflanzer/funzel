// No pragma once, we want it multiple times!

#include <gtest/gtest.h>
#include <funzel/Tensor.hpp>
#include <funzel/nn/NNBackendTensor.hpp>
#include <funzel/cv/CVBackendTensor.hpp>

#include <cmath>

using namespace funzel;

#ifndef CommonTest
#define CommonTest DefaultTest
#endif

#ifndef TestDevice
#define TestDevice ""
#endif

#define CommonTestNN CAT(CommonTest, _NN)

#include <funzel/nn/Linear.hpp>
TEST(CommonTestNN, LinearLayer)
{
	nn::Linear lin(3, 9);

	lin.bias().fill(1);
	lin.weights().fill(1);

	lin.to(TestDevice);

	auto v = Tensor::ones({5, 3}).to(TestDevice);
	auto r = lin(v).cpu();

	Tensor expected({5, 9},
	{
		4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f,
		4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f,
		4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f,
		4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f,
		4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f
	});

	EXPECT_EQ(r.shape, (Shape{5, 9}));
	EXPECT_TENSOR_EQ(r, expected);

	std::cout << r << std::endl;
}

#include <funzel/nn/Sequential.hpp>
#include <funzel/nn/Sigmoid.hpp>

TEST(CommonTestNN, SequentialLayer)
{
	using namespace nn;
	Sequential seq{
		std::make_shared<Linear>(2, 8),
		std::make_shared<Sigmoid>(),

		std::make_shared<Linear>(8, 16),
		std::make_shared<Sigmoid>(),

		std::make_shared<Linear>(16, 32),
		std::make_shared<Sigmoid>(),

		std::make_shared<Linear>(32, 1024),
		std::make_shared<Sigmoid>()
	};

	seq.defaultInitialize();
	seq.to(TestDevice);

	auto in = Tensor::ones({10, 2}, DFLOAT32, TestDevice);
	auto r = seq(in).cpu();

	EXPECT_EQ(r.shape, (Shape{10, 1024}));
	// TODO Check values!
}

#if 0
TEST(CommonTestNN, Pool2D)
{
	Tensor a = funzel::linspace(1, 256*256, 256*256).reshape({1, 256, 256}).to(TestDevice);

	UVec2 padding{0, 0}, kernelSize{2, 2}, stride{2, 2}, dilation{1, 1};

	size_t width = ((a.shape[1] + 2*padding[0] - dilation[0]*(kernelSize[0] - 1) - 1)/stride[0]) + 1;
	size_t height = ((a.shape[2] + 2*padding[1] - dilation[1]*(kernelSize[1] - 1) - 1)/stride[1]) + 1;

	Tensor b = Tensor::empty({1, width, height}, DFLOAT32, TestDevice);

	std::cout << a.cpu() << std::endl;

	a.getBackendAs<nn::NNBackendTensor>()->pool2d(a, b, MAX_POOLING, kernelSize, stride, padding, dilation);
	std::cout << b.cpu() << std::endl;

	a.getBackendAs<nn::NNBackendTensor>()->pool2d(a, b, MEAN_POOLING, kernelSize, stride, padding, dilation);
	std::cout << b.cpu() << std::endl;
}
#endif

TEST(CommonTestNN, ReLU)
{
	Tensor v = Tensor::empty({3, 3, 3});
	funzel::randn(v);

	v = v.to(TestDevice);
	v.getBackendAs<nn::NNBackendTensor>()->relu(v, v, 0);
	v = v.cpu();

	std::cout << v << std::endl;

#if 0
	for(size_t p = 0; p < 3; p++)
		for(size_t q = 0; q < 3; q++)
			for(size_t r = 0; r < 3; r++)
			{
				EXPECT_EQ((v[{p, q, r}].item<float>()), std::max());
			}
#endif
}

#undef CommonTestNN
