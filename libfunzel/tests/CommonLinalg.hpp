// No pragma once, we want it multiple times!

#include <gtest/gtest.h>
#include <funzel/Tensor.hpp>
#include <funzel/linalg/LinalgBackendTensor.hpp>
#include <funzel/linalg/Linalg.hpp>

#include <cmath>

using namespace funzel;

#ifndef CommonTest
#define CommonTest DefaultTest
#endif

#ifndef TestDevice
#define TestDevice ""
#endif

#define CommonTestLinalg CAT(CommonTest, _Linalg)

TEST(CommonTestLinalg, Det)
{
	Tensor a({3, 3},
	{
		2.0f, 0.0f, 0.0f,
		0.0f, 5.0f, 0.0f,
		0.0f, 0.0f, 2.0f
	}, TestDevice);

	const auto tgt = linalg::det(a);
	EXPECT_FLOAT_EQ(tgt.item<float>(), 20.0f);
}

TEST(CommonTestLinalg, DetBroadcast)
{
	Tensor a({4, 3, 3},
	{
		2.0f, 0.0f, 0.0f,
		0.0f, 5.0f, 0.0f,
		0.0f, 0.0f, 2.0f,

		2.0f, 0.0f, 0.0f,
		0.0f, 5.0f, 0.0f,
		0.0f, 0.0f, 2.0f,

		2.0f, 0.0f, 0.0f,
		0.0f, 5.0f, 0.0f,
		0.0f, 0.0f, 2.0f,

		2.0f, 0.0f, 0.0f,
		0.0f, 5.0f, 0.0f,
		0.0f, 0.0f, 2.0f,
	}, TestDevice);

	const auto tgt = linalg::det(a);
	const Tensor expectedResult({4}, {20.0f, 20.0f, 20.0f, 20.0f});
	EXPECT_TENSOR_EQ(tgt.cpu(), expectedResult);
}

#undef CommonTestLinalg
