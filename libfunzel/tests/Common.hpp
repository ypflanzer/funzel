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

TEST(CommonTest, Sum)
{
	auto v = Tensor::ones({3, 3, 3});
	EXPECT_EQ(v.sum(), 3*3*3);
	EXPECT_EQ(v[0].sum(), 3*3);
	
	v[{0, 0, 1}] = 0;

	EXPECT_EQ(v.transpose()[0].sum(), 3*3);
	EXPECT_EQ(v[0].sum(), 3*3 - 1);

	double overall = 0;
	for(size_t p = 0; p < 3; p++)
		for(size_t q = 0; q < 3; q++)
			for(size_t r = 0; r < 3; r++)
			{
				auto x = rand() % 10;
				overall += x;
				v[{p, q, r}] = x;
			}

	EXPECT_EQ(v.transpose().sum(), overall);
}

TEST(CommonTest, Abs)
{
	auto v = Tensor::empty({3, 3, 3});
	for(size_t p = 0; p < 3; p++)
		for(size_t q = 0; q < 3; q++)
			for(size_t r = 0; r < 3; r++)
			{
				v[{p, q, r}] = -1;
			}

	v = v.to(TestDevice);
	v.abs_();
	v = v.cpu();

	for(size_t p = 0; p < 3; p++)
		for(size_t q = 0; q < 3; q++)
			for(size_t r = 0; r < 3; r++)
			{
				EXPECT_EQ((v[{p, q, r}].item<float>()), 1);
			}
}

TEST(CommonTest, MulScalar)
{
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice);
	v.mul_(32);
	v = v.cpu();

	for(size_t p = 0; p < 3; p++)
		for(size_t q = 0; q < 3; q++)
			for(size_t r = 0; r < 3; r++)
			{
				EXPECT_EQ((v[{p, q, r}].item<float>()), 32);
			}
}

TEST(CommonTest, Exp)
{
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice);
	v.mul_(2);
	v.exp_();
	v = v.cpu();

	for(size_t p = 0; p < 3; p++)
		for(size_t q = 0; q < 3; q++)
			for(size_t r = 0; r < 3; r++)
			{
				EXPECT_FLOAT_EQ((v[{p, q, r}].item<float>()), std::exp(2));
			}
}

TEST(CommonTest, Sqrt)
{
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice);
	v.mul_(2);
	v.sqrt_();
	v = v.cpu();

	for(size_t p = 0; p < 3; p++)
		for(size_t q = 0; q < 3; q++)
			for(size_t r = 0; r < 3; r++)
			{
				EXPECT_FLOAT_EQ((v[{p, q, r}].item<float>()), std::sqrt(2));
			}
}

TEST(CommonTest, Sin)
{
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice);
	v.mul_(2);
	v.sin_();
	v = v.cpu();

	for(size_t p = 0; p < 3; p++)
		for(size_t q = 0; q < 3; q++)
			for(size_t r = 0; r < 3; r++)
			{
				EXPECT_FLOAT_EQ((v[{p, q, r}].item<float>()), std::sin(2));
			}
}

TEST(CommonTest, Cos)
{
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice);
	v.mul_(2);
	v.cos_();
	v = v.cpu();

	for(size_t p = 0; p < 3; p++)
		for(size_t q = 0; q < 3; q++)
			for(size_t r = 0; r < 3; r++)
			{
				EXPECT_FLOAT_EQ((v[{p, q, r}].item<float>()), std::cos(2));
			}
}

#if 0
TEST(CommonTest, Sigmoid)
{
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice);
	v.mul_(2);
	v.sigmoid_();
	v = v.cpu();

	for(size_t p = 0; p < 3; p++)
		for(size_t q = 0; q < 3; q++)
			for(size_t r = 0; r < 3; r++)
			{
				EXPECT_FLOAT_EQ((v[{p, q, r}].item<float>()), 0.88079707797788);
			}
}
#endif

TEST(CommonTest, Tan)
{
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice);
	v.mul_(2);
	v.tan_();
	v = v.cpu();

	for(size_t p = 0; p < 3; p++)
		for(size_t q = 0; q < 3; q++)
			for(size_t r = 0; r < 3; r++)
			{
				EXPECT_FLOAT_EQ((v[{p, q, r}].item<float>()), std::tan(2));
			}
}

TEST(CommonTest, Tanh)
{
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice);
	v.mul_(2);
	v.tanh_();
	v = v.cpu();

	for(size_t p = 0; p < 3; p++)
		for(size_t q = 0; q < 3; q++)
			for(size_t r = 0; r < 3; r++)
			{
				EXPECT_FLOAT_EQ((v[{p, q, r}].item<float>()), std::tanh(2));
			}
}

TEST(CommonTest, AddMatrix)
{
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice);
	v.add_(v);
	v = v.cpu();

	for(size_t p = 0; p < 3; p++)
		for(size_t q = 0; q < 3; q++)
			for(size_t r = 0; r < 3; r++)
			{
				EXPECT_EQ((v[{p, q, r}].item<float>()), 2);
			}
}

#define EXPECT_TENSOR_EQ(t1, t2) \
ASSERT_EQ((t1).shape, (t2).shape); \
ASSERT_EQ((t1).dtype, (t2).dtype); \
	for(size_t i = 0; i < (t1).size(); i++) \
	{ \
		float* cdata = (float*) (t1).data(i*sizeof(float)); \
		float* edata = (float*) (t2).data(i*sizeof(float)); \
		EXPECT_EQ(*cdata, *edata); \
	}

TEST(CommonTest, MatmulTensor)
{
	Tensor a({2, 3, 3},
	{
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f,
		7.0f, 8.0f, 9.0f,

		9.0f, 8.0f, 7.0f,
		6.0f, 5.0f, 4.0f,
		3.0f, 2.0f, 1.0f
	}, TestDevice);

	Tensor b({2, 3, 3},
	{
		9.0f, 8.0f, 7.0f,
		6.0f, 5.0f, 4.0f,
		3.0f, 2.0f, 1.0f,

		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f,
		7.0f, 8.0f, 9.0f
	}, TestDevice);

	Tensor expected({2, 3, 3},
	{
		30.0f, 24.0f, 18.0f,
		84.0f, 69.0f, 54.0f,
		138.0f, 114.0f, 90.0f,

		90.0f, 114.0f, 138.0f,
		54.0f, 69.0f, 84.0f,
		18.0f, 24.0f, 30.0f,
	});

	auto c = a.matmul(b).cpu();
	EXPECT_TENSOR_EQ(c, expected);
}

TEST(CommonTest, Matmul)
{
	Tensor a({2, 3},
	{
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f,
	}, TestDevice);

	Tensor b({3, 2},
	{
		9.0f, 8.0f, 7.0f,
		6.0f, 5.0f, 4.0f,
	}, TestDevice);

	Tensor expected({2, 2},
	{
		38.0f,  32.0f,
		101.0f, 86.0f
	});

	auto c = a.matmul(b).cpu();
	EXPECT_TENSOR_EQ(c, expected);
}

TEST(CommonTest, MatmulMatrixVector)
{
	Tensor a({3, 3},
	{
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f,
		7.0f, 8.0f, 9.0f
	}, TestDevice);

	Tensor b({3, 1},
	{
		0.0f, 1.0f, 0.0f
	}, TestDevice);

	Tensor expected({3, 1},
	{
		2.0f, 5.0f, 8.0f
	});

	auto c = a.matmul(b).cpu();
	EXPECT_TENSOR_EQ(c, expected);
}

TEST(CommonTest, MatmulVectorVector)
{
	Tensor a({3, 1},
	{
		1.0f, 2.0f, 3.0f,
	}, TestDevice);

	Tensor b({3, 1},
	{
		1.0f, 1.0f, 1.0f
	}, TestDevice);

	Tensor expected({1},
	{
		6.0f
	});

	auto c = a.transpose().matmul(b).cpu();
	EXPECT_TENSOR_EQ(c, expected);
}

TEST(CommonTest, BroadcastVectorVectorMul)
{
	Tensor a({4, 1, 3},
	{
		1.0f, 2.0f, 3.0f,
		1.0f, 2.0f, 3.0f,
		1.0f, 2.0f, 3.0f,
		1.0f, 2.0f, 3.0f,
	}, TestDevice);

	Tensor b({3, 1},
	{
		1.0f, 1.0f, 1.0f
	}, TestDevice);

	Tensor expected({4, 1},
	{
		6.0f, 6.0f, 6.0f, 6.0f
	});

	auto c = a.matmul(b).cpu();
	std::cout << c << std::endl;
	EXPECT_TENSOR_EQ(c, expected);
}

TEST(CommonTest, BroadcastVectorVectorAdd)
{
	Tensor a({4, 3},
	{
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f,
		7.0f, 8.0f, 9.0f,
		10.0f, 11.0f, 12.0f,
	}, TestDevice);

	Tensor b({3},
	{
		1.0f, 1.0f, 1.0f
	}, TestDevice);

	Tensor expected({4, 3},
	{
		2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f,
		8.0f, 9.0f, 10.0f,
		11.0f, 12.0f, 13.0f,
	});

	auto c = a.add(b).cpu();
	EXPECT_TENSOR_EQ(c, expected);
}

TEST(CommonTest, BroadcastVectorVectorDiv)
{
	Tensor a({4, 3},
	{
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f,
		7.0f, 8.0f, 9.0f,
		10.0f, 11.0f, 12.0f,
	}, TestDevice);

	Tensor b({3},
	{
		2.0f, 2.0f, 2.0f
	}, TestDevice);

	Tensor expected({4, 3},
	{
		1.0f/2.0f, 2.0f/2.0f, 3.0f/2.0f,
		4.0f/2.0f, 5.0f/2.0f, 6.0f/2.0f,
		7.0f/2.0f, 8.0f/2.0f, 9.0f/2.0f,
		10.0f/2.0f, 11.0f/2.0f, 12.0f/2.0f,
	});

	auto c = a.div(b).cpu();
	EXPECT_TENSOR_EQ(c, expected);
}

#include <funzel/nn/Linear.hpp>
TEST(CommonTest, LinearLayer)
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

TEST(CommonTest, SequentialLayer)
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

	auto in = Tensor::ones({10, 2}, FLOAT32, TestDevice);
	auto r = seq(in).cpu();

	EXPECT_EQ(r.shape, (Shape{10, 1024}));
	// TODO Check values!
}

TEST(CommonTest, Pool2D)
{
	Tensor a = funzel::linspace(1, 256*256, 256*256).reshape({1, 256, 256}).to(TestDevice);

	UVec2 padding{0, 0}, kernelSize{2, 2}, stride{2, 2}, dilation{1, 1};

	size_t width = ((a.shape[1] + 2*padding[0] - dilation[0]*(kernelSize[0] - 1) - 1)/stride[0]) + 1;
	size_t height = ((a.shape[2] + 2*padding[1] - dilation[1]*(kernelSize[1] - 1) - 1)/stride[1]) + 1;

	Tensor b = Tensor::empty({1, width, height}, FLOAT32, TestDevice);

	std::cout << a.cpu() << std::endl;

	a.getBackendAs<nn::NNBackendTensor>()->pool2d(a, b, MAX_POOLING, kernelSize, stride, padding, dilation);
	std::cout << b.cpu() << std::endl;

	a.getBackendAs<nn::NNBackendTensor>()->pool2d(a, b, MEAN_POOLING, kernelSize, stride, padding, dilation);
	std::cout << b.cpu() << std::endl;
}

#include <funzel/cv/Image.hpp>
#include <funzel/Plot.hpp>

TEST(CommonTest, Conv2d)
{
	auto img = image::load("mnist_png/training/8/225.png").astype<float>().to(TestDevice).mul_(1.0 / 255.0);
	img.shape.erase(img.shape.begin() + 2);
	img.reshape_(img.shape);

	auto tgt = Tensor::zeros_like(img);
	
#if 1
	auto kernel = Tensor::ones({ 5, 5 }, FLOAT32, TestDevice);
	kernel.mul_(1.0 / (5.0*5.0));
	img.getBackendAs<cv::CVBackendTensor>()->conv2d(img, tgt, kernel, { 1, 1 }, { 2, 2 }, { 1, 1 });
#else
	Tensor kernel({ 5, 5 }, {
		0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, -1.0f, 0.0f,
		0.0f, 1.0f, 0.0f, -1.0f, 0.0f,
		0.0f, 1.0f, 0.0f, -1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
	}, TestDevice);
	kernel.mul_(1.0 / 3.0);
	img.getBackendAs<nn::NNBackendTensor>()->conv2d(img, tgt, kernel, { 1, 1 }, { 2, 2 }, { 1, 1 });
#endif

	//tgt.shape.push_back(1);
	//tgt.permute_({1, 2, 0});

	tgt = tgt.cpu().mul_(255.0).astype<uint8_t>();
	image::save(tgt, "CommonTest_Conv2d.png");

#if 1
	Plot plt;
	//plt.image(img.mul(255.0).cpu().astype<uint8_t>());
	plt.image(tgt, "Result");
	plt.show();
#endif
}

TEST(CommonTest, Conv2dColor)
{
	auto img = image::load("color_image.png", image::CHW).astype<float>().to(TestDevice).mul_(1.0 / 255.0);
	auto tgt = Tensor::zeros_like(img);
	
	auto kernel = Tensor::ones({ 5, 5 }, FLOAT32, TestDevice);
	kernel.mul_(1.0 / (5.0*5.0));
	img.getBackendAs<cv::CVBackendTensor>()->conv2d(img, tgt, kernel, { 1, 1 }, { 2, 2 }, { 1, 1 });

	tgt = tgt.cpu().mul_(255.0).astype<uint8_t>();
	tgt = image::toOrder(tgt, image::HWC);
	image::save(tgt, "CommonTest_Conv2d.png");

#if 1
	Plot plt;
	//plt.image(img.mul(255.0).cpu().astype<uint8_t>());
	plt.image(tgt, "Result");
	plt.show();
#endif
}

TEST(CommonTest, ConvertToGrayscale)
{
	auto img = image::load("test.jpg", image::CHW).astype<float>().to(TestDevice);
	auto tgt = Tensor::empty({1, img.shape[1], img.shape[2]}, img.dtype);

	img.getBackendAs<cv::CVBackendTensor>()->convertGrayscale(img, tgt);

	tgt = tgt.cpu().astype<uint8_t>();
	tgt = image::toOrder(tgt, image::HWC);
	image::save(tgt, "CommonTest_Grayscale.png");
	image::imshow(tgt, "Grayscale", true);
}

TEST(CommonTest, ReLU)
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
