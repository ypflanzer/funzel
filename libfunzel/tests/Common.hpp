// No pragma once, we want it multiple times!

#include <gtest/gtest.h>
#include <funzel/Tensor.hpp>
#include <funzel/nn/NNBackendTensor.hpp>
#include <funzel/cv/CVBackendTensor.hpp>

#include <cmath>

using namespace funzel;

// For all following test suites
#define STRINGIFY2( x) #x
#define STRINGIFY(x) STRINGIFY2(x)

#define CAT_(x, y) x ## y
#define CAT(x, y) CAT_(x, y)

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

TEST(CommonTest, AddMatrixStrided)
{
	const auto ones = Tensor::ones({3, 3}).to(TestDevice);
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice).transpose();
	v[1].add_(ones);
	v = v.transpose();
	v = v.cpu();

	for(size_t p = 0; p < 3; p++)
		for(size_t q = 0; q < 3; q++)
			for(size_t r = 0; r < 3; r++)
			{
				EXPECT_EQ((v[{p, q, r}].item<float>()), (r == 1 ? 2 : 1));
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
	Tensor a({1, 3},
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

	auto c = a.matmul(b).cpu();
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
