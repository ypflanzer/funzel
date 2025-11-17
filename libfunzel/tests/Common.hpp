// No pragma once, we want it multiple times!

#include <gtest/gtest.h>
#include <funzel/Tensor.hpp>
#include <funzel/nn/NNBackendTensor.hpp>
#include <funzel/cv/CVBackendTensor.hpp>

#include "TestUtils.hpp"

#include <cmath>

#include <spdlog/spdlog.h>

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

#include "BasicOpsTest.hpp"

TEST(CommonTest, Fill)
{
	auto v = Tensor::empty({3, 3, 3}, funzel::DFLOAT32, TestDevice);
	v.fill(42);
	v = v.cpu();

	for(int64_t p = 0; p < 3; p++)
		for(int64_t q = 0; q < 3; q++)
			for(int64_t r = 0; r < 3; r++)
			{
				EXPECT_EQ((v[{p, q, r}].item<float>()), 42);
			}
}

TEST(CommonTest, FillWithStrides)
{
	auto v = Tensor::empty({3, 3, 3}, funzel::DFLOAT32, TestDevice);
	v.fill(42);

	// Test with default strides
	auto v_cpu = v.cpu();
	for(int64_t p = 0; p < 3; p++)
		for(int64_t q = 0; q < 3; q++)
			for(int64_t r = 0; r < 3; r++)
				EXPECT_EQ((v_cpu[{p, q, r}].item<float>()), 42);

	// Test with transposed strides
	auto v_transposed = v.transpose().cpu();
	for(int64_t p = 0; p < 3; p++)
		for(int64_t q = 0; q < 3; q++)
			for(int64_t r = 0; r < 3; r++)
				EXPECT_EQ((v_transposed[{p, q, r}].item<float>()), 42);

	// Test with permuted strides (swap axes 0 and 2)
	auto v_permuted = v.permute({2, 1, 0}).cpu();
	for(int64_t p = 0; p < 3; p++)
		for(int64_t q = 0; q < 3; q++)
			for(int64_t r = 0; r < 3; r++)
				EXPECT_EQ((v_permuted[{p, q, r}].item<float>()), 42);
}

TEST(CommonTest, FillWithSlice)
{
	auto v = Tensor::zeros({3, 3, 3}, funzel::DFLOAT32, TestDevice);

	// Fill a slice (all elements at r%2 = 0)
	v.slice({{}, {}, {0, -1, 2}}).fill(42);

	// Test that only the slice is filled
	auto v_cpu = v.cpu();
	for(int64_t p = 0; p < 3; p++)
		for(int64_t q = 0; q < 3; q++)
			for(int64_t r = 0; r < 3; r++)
			{
				if(r == 0)
					EXPECT_EQ((v_cpu[{p, q, r}].item<float>()), 42);
				else
					EXPECT_EQ((v_cpu[{p, q, r}].item<float>()), 0);
			}
}

TEST(CommonTest, Sum)
{
	auto v = Tensor::ones({3, 3, 3});

	// Test simple sum of all elements
	EXPECT_EQ(v.sum().item<float>(), 3*3*3);

	// Test sum for one submatrix
	EXPECT_EQ(v[0].sum().item<float>(), 3*3);
}

TEST(CommonTest, SumStrided)
{
	auto v = Tensor::ones({3, 3, 3});
	
	// Set one value to zero such that a transpose will show different results
	v[{0, 0, 1}] = 0;

	// This tests whether strides work
	EXPECT_EQ(v.transpose()[0].sum().item<float>(), 3*3 - 1);
	EXPECT_EQ(v[0].sum().item<float>(), 3*3 - 1);

	// Set random values and keep sum
	double overall = 0;
	for(int64_t p = 0; p < 3; p++)
		for(int64_t q = 0; q < 3; q++)
			for(int64_t r = 0; r < 3; r++)
			{
				auto x = rand() % 10;
				overall += x;
				v[{p, q, r}] = x;
			}

	// Check if a strided version works
	EXPECT_EQ(v.transpose().sum().item<float>(), overall);
}

TEST(CommonTest, Sum3D)
{
	auto tensor = Tensor::ones({1, 2, 3}, funzel::DFLOAT32, TestDevice);

	// Scalar
	if(false)
	{
		auto scalarSum = tensor.sum();
		const Tensor scalarSumExpected({1}, {6.0f});
		EXPECT_TENSOR_EQ(scalarSum.cpu(), scalarSumExpected);
	}

	// Axis 0
	{
		auto sum = tensor.sum({0});
		std::cout<< "SUM: " << sum << std::endl;
		const Tensor sumExpected({2, 3}, {
			1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f
		});
		
		EXPECT_TENSOR_EQ(sum.cpu(), sumExpected);
	}

	return;

	// Axis 1
	{
		auto sum = tensor.sum({1});
		const Tensor sumExpected({2}, {3.0f, 3.0f});
		EXPECT_TENSOR_EQ(sum.cpu(), sumExpected);
	}
}

TEST(CommonTest, Abs)
{
	auto v = Tensor::empty({3, 3, 3});
	for(int64_t p = 0; p < 3; p++)
		for(int64_t q = 0; q < 3; q++)
			for(int64_t r = 0; r < 3; r++)
			{
				v[{p, q, r}] = -1;
			}

	v = v.to(TestDevice);
	v.abs_();
	v = v.cpu();

	for(int64_t p = 0; p < 3; p++)
		for(int64_t q = 0; q < 3; q++)
			for(int64_t r = 0; r < 3; r++)
			{
				EXPECT_EQ((v[{p, q, r}].item<float>()), 1);
			}
}

TEST(CommonTest, AbsStrided)
{
	auto v = Tensor::ones({3, 3, 3}).mul(-1).to(TestDevice).transpose();
	v[1].abs_();
	v = v.transpose();
	v = v.cpu();

	for(int64_t p = 0; p < 3; p++)
		for(int64_t q = 0; q < 3; q++)
			for(int64_t r = 0; r < 3; r++)
			{
				EXPECT_FLOAT_EQ((v[{p, q, r}].item<float>()), (r == 1 ? 1 : -1));
			}
}

TEST(CommonTest, MulScalar)
{
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice);
	v.mul_(32);
	v = v.cpu();

	for(int64_t p = 0; p < 3; p++)
		for(int64_t q = 0; q < 3; q++)
			for(int64_t r = 0; r < 3; r++)
			{
				EXPECT_EQ((v[{p, q, r}].item<float>()), 32);
			}
}

TEST(CommonTest, MulScalarStrided)
{
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice).transpose();
	v[1].mul_(32);
	v = v.transpose();
	v = v.cpu();

	for(int64_t p = 0; p < 3; p++)
		for(int64_t q = 0; q < 3; q++)
			for(int64_t r = 0; r < 3; r++)
			{
				const auto value = v[{p, q, r}].item<float>();
				EXPECT_FLOAT_EQ((value), (r == 1 ? 32 : 1));
			}
}

TEST(CommonTest, Exp)
{
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice);
	v.mul_(2);
	v.exp_();
	v = v.cpu();

	for(int64_t p = 0; p < 3; p++)
		for(int64_t q = 0; q < 3; q++)
			for(int64_t r = 0; r < 3; r++)
			{
				EXPECT_FLOAT_EQ((v[{p, q, r}].item<float>()), std::exp(2));
			}
}

TEST(CommonTest, ExpStrided)
{
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice).mul_(2).transpose();
	v[1].exp_();
	v = v.transpose();
	v = v.cpu();

	for(int64_t p = 0; p < 3; p++)
		for(int64_t q = 0; q < 3; q++)
			for(int64_t r = 0; r < 3; r++)
			{
				const auto value = v[{p, q, r}].item<float>();
				EXPECT_FLOAT_EQ(value, (r == 1 ? std::exp(2) : 2));
			}
}

TEST(CommonTest, Sqrt)
{
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice);
	v.mul_(2);
	v.sqrt_();
	v = v.cpu();

	for(int64_t p = 0; p < 3; p++)
		for(int64_t q = 0; q < 3; q++)
			for(int64_t r = 0; r < 3; r++)
			{
				EXPECT_FLOAT_EQ((v[{p, q, r}].item<float>()), std::sqrt(2));
			}
}

TEST(CommonTest, SqrtStrided)
{
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice).mul_(2).transpose();
	v[1].sqrt_();
	v = v.transpose();
	v = v.cpu();

	for(int64_t p = 0; p < 3; p++)
		for(int64_t q = 0; q < 3; q++)
			for(int64_t r = 0; r < 3; r++)
			{
				EXPECT_FLOAT_EQ((v[{p, q, r}].item<float>()), (r == 1 ? std::sqrt(2) : 2));
			}
}

TEST(CommonTest, Sin)
{
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice);
	v.mul_(2);
	v.sin_();
	v = v.cpu();

	for(int64_t p = 0; p < 3; p++)
		for(int64_t q = 0; q < 3; q++)
			for(int64_t r = 0; r < 3; r++)
			{
				EXPECT_FLOAT_EQ((v[{p, q, r}].item<float>()), std::sin(2));
			}
}

TEST(CommonTest, SinStrided)
{
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice).mul_(2).transpose();
	v[1].sin_();
	v = v.transpose();
	v = v.cpu();

	for(int64_t p = 0; p < 3; p++)
		for(int64_t q = 0; q < 3; q++)
			for(int64_t r = 0; r < 3; r++)
			{
				EXPECT_FLOAT_EQ((v[{p, q, r}].item<float>()), (r == 1 ? std::sin(2) : 2));
			}
}

TEST(CommonTest, Cos)
{
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice);
	v.mul_(2);
	v.cos_();
	v = v.cpu();

	for(int64_t p = 0; p < 3; p++)
		for(int64_t q = 0; q < 3; q++)
			for(int64_t r = 0; r < 3; r++)
			{
				EXPECT_FLOAT_EQ((v[{p, q, r}].item<float>()), std::cos(2));
			}
}

TEST(CommonTest, CosStrided)
{
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice).mul_(2).transpose();
	v[1].cos_();
	v = v.transpose();
	v = v.cpu();

	for(int64_t p = 0; p < 3; p++)
		for(int64_t q = 0; q < 3; q++)
			for(int64_t r = 0; r < 3; r++)
			{
				EXPECT_FLOAT_EQ((v[{p, q, r}].item<float>()), (r == 1 ? std::cos(2) : 2));
			}
}

#if 0
TEST(CommonTest, Sigmoid)
{
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice);
	v.mul_(2);
	v.sigmoid_();
	v = v.cpu();

	for(int64_t p = 0; p < 3; p++)
		for(int64_t q = 0; q < 3; q++)
			for(int64_t r = 0; r < 3; r++)
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

	for(int64_t p = 0; p < 3; p++)
		for(int64_t q = 0; q < 3; q++)
			for(int64_t r = 0; r < 3; r++)
			{
				EXPECT_FLOAT_EQ((v[{p, q, r}].item<float>()), std::tan(2));
			}
}

TEST(CommonTest, TanStrided)
{
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice).mul_(2).transpose();
	v[1].tan_();
	v = v.transpose();
	v = v.cpu();

	for(int64_t p = 0; p < 3; p++)
		for(int64_t q = 0; q < 3; q++)
			for(int64_t r = 0; r < 3; r++)
			{
				EXPECT_FLOAT_EQ((v[{p, q, r}].item<float>()), (r == 1 ? std::tan(2) : 2));
			}
}

TEST(CommonTest, Tanh)
{
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice);
	v.mul_(2);
	v.tanh_();
	v = v.cpu();

	for(int64_t p = 0; p < 3; p++)
		for(int64_t q = 0; q < 3; q++)
			for(int64_t r = 0; r < 3; r++)
			{
				EXPECT_FLOAT_EQ((v[{p, q, r}].item<float>()), std::tanh(2));
			}
}

TEST(CommonTest, TanhStrided)
{
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice).mul_(2).transpose();
	v[1].tanh_();
	v = v.transpose();
	v = v.cpu();

	for(int64_t p = 0; p < 3; p++)
		for(int64_t q = 0; q < 3; q++)
			for(int64_t r = 0; r < 3; r++)
			{
				EXPECT_FLOAT_EQ((v[{p, q, r}].item<float>()), (r == 1 ? std::tanh(2) : 2));
			}
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

TEST(CommonTest, FuncMean1D)
{
	Tensor a({9}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}, TestDevice);
	auto c = funzel::mean(a);
	
	const Tensor expected({1}, {5.0f}, TestDevice);
	EXPECT_TENSOR_EQ(c, expected);
}

TEST(CommonTest, FuncMean2D)
{
	Tensor a({3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}, TestDevice);
	
	auto c = funzel::mean(a);
	
	const Tensor expected({1}, {5.0f}, TestDevice);
	EXPECT_TENSOR_EQ(c, expected);
}

TEST(CommonTest, FuncMeanAxis2D)
{
	Tensor a({3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}, TestDevice);
	Tensor mean = funzel::mean(a, {1});

	Tensor expected({3},
	{
		2.0f, 5.0f, 8.0f
	});

	EXPECT_TENSOR_EQ(mean.cpu(), expected);
}

TEST(CommonTest, Mean1D)
{
	Tensor a({9}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}, TestDevice);
	auto c = a.mean();
	
	const Tensor expected({1}, {5.0f}, TestDevice);
	EXPECT_TENSOR_EQ(c, expected);
}

TEST(CommonTest, Mean2D)
{
	Tensor a({3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}, TestDevice);
	
	auto c = a.mean();
	
	const Tensor expected({1}, {5.0f}, TestDevice);
	EXPECT_TENSOR_EQ(c, expected);
}

TEST(CommonTest, MeanAxis2D)
{
	Tensor a({3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}, TestDevice);
	Tensor mean = a.mean({1});

	Tensor expected({3},
	{
		2.0f, 5.0f, 8.0f
	});

	EXPECT_TENSOR_EQ(mean.cpu(), expected);
}
