#include <gtest/gtest.h>
#include <funzel/Tensor.hpp>
#include <funzel/nn/NNBackendTensor.hpp>

#include "../../TestUtils.hpp"

using namespace funzel;

TEST(CPUTensorTest, PrintDevices)
{
	funzel::PrintDevices();
}

TEST(CPUTensorTest, Init)
{
	auto ones = Tensor::ones({3, 3, 3});
	EXPECT_EQ(ones.strides, Strides({36, 12, 4}));
	EXPECT_EQ(ones.shape, Shape({3, 3, 3}));
	EXPECT_EQ(ones->size, 3*3*3*sizeof(float));
	EXPECT_EQ(ones.dtype, DTYPE::DFLOAT32);
}

TEST(CPUTensorTest, ToString)
{
	auto ones = Tensor::ones({20, 1});
	std::cout << ones << std::endl;

	ones = Tensor::ones({20, 20});
	std::cout << ones << std::endl;
}

TEST(CPUTensorTest, Transpose)
{
	auto v = Tensor::zeros({3, 3});

	v[{1, 0}] = 1;

	EXPECT_EQ((v[{1, 0}].item<float>()), 1.0f);
	EXPECT_EQ(v.strides, Strides({12, 4}));
	EXPECT_EQ(v.shape, Shape({3, 3}));

	v = v.transpose();

	EXPECT_EQ((v[{0, 1}].item<float>()), 1.0f);
	EXPECT_EQ(v.strides, Strides({4, 12}));
	EXPECT_EQ(v.shape, Shape({3, 3}));
}

TEST(CPUTensorTest, Index)
{
	auto ones = Tensor::ones({3, 2, 1}, DTYPE::DFLOAT32);

	EXPECT_EQ(0, ones.offset);
	EXPECT_EQ(0, ones[0].offset);

	EXPECT_EQ(2*sizeof(float), ones[1].offset);
	EXPECT_EQ(ones[0].shape[0], 2);
}

TEST(CPUTensorTest, IndexNegativeIndex)
{
	auto ones = Tensor::ones({3, 2, 1}, DTYPE::DFLOAT32);

	EXPECT_EQ(0, ones.offset);
	EXPECT_EQ((3*2 - 2)*sizeof(float), ones[-1].offset);
}

TEST(CPUTensorTest, Unravel)
{
	auto v = Tensor::zeros({3, 3});

	v[{1, 0}] = 1;
	EXPECT_EQ(v.dataAs<float>(3), 1);

	auto vtrans = v.transpose();
	EXPECT_TRUE(!vtrans.isContiguous());

	// The physical location of the value should not have changed
	EXPECT_EQ(v.dataAs<float>(3), 1);

	vtrans = vtrans.unravel();
	EXPECT_TRUE(vtrans.isContiguous());

	// Now it should have.
	EXPECT_EQ(vtrans.dataAs<float>(1), 1);
}

TEST(CPUTensorTest, Reshape)
{
	auto v = Tensor::zeros({3, 3});
	v[{1, 0}] = 1;

	EXPECT_THROW(v.reshape({27, 2}), std::runtime_error);

	v.reshape_({9});

	EXPECT_EQ(v.shape, Shape{9});
}

TEST(CPUTensorTest, Permute)
{
	auto v = Tensor::ones({3, 4, 5});
	v.permute_({2, 0, 1});

	std::cout << v.shape << v.strides << std::endl;
}

TEST(CPUTensorTest, Assign)
{
	Tensor a = Tensor::zeros({3, 3});
	Tensor b = Tensor::ones(3);

	Tensor subset = a[1];
	a[1].set(b);
	
	std::cout << a.cpu() << std::endl;
}

TEST(CPUTensorTest, Linspace)
{
	Tensor a = funzel::linspace(Tensor({3}, {0.0f, 0.0f, 0.0f}), Tensor({3}, {1.0f, 2.0f, 3.0f}), 50);
	std::cout << a << std::endl;
}

TEST(CPUTensorTest, TrimDimensions)
{
	Tensor a = Tensor::empty({3, 1, 1, 3, 1, 1});

	a.trimDimensions();
	EXPECT_EQ(a.shape, (Shape({3, 1, 3, 1})));
}

TEST(CPUTensorTest, Flatten)
{
	Tensor a = Tensor::ones({3, 1, 1, 3, 1, 1});
	Tensor b = a.flatten();

	EXPECT_EQ(b.shape, (Shape({9})));
}

TEST(CPUTensorTest, Slice)
{
	Tensor a = Tensor::ones({5, 5, 5});

	auto sliced = a({{1,4}, {}, {}}).fill(2.0);

	EXPECT_EQ(sliced.shape, (Shape{3, 5, 5}));
	EXPECT_EQ(sliced.strides, (Strides{100, 20, 4}));

	auto twos = Tensor::empty_like(sliced).fill(2.0);
	EXPECT_TENSOR_EQ(sliced, twos);
}

TEST(CPUTensorTest, SliceNegative)
{
	Tensor a = Tensor::ones({5, 5, 5});

	auto sliced = a({{0,-4}, {}, {}}).fill(2.0);

	EXPECT_EQ(sliced.shape, (Shape{1, 5, 5}));
	EXPECT_EQ(sliced.strides, (Strides{100, 20, 4}));

	auto twos = Tensor::empty_like(sliced).fill(2.0);
	EXPECT_TENSOR_EQ(sliced, twos);
}

TEST(CPUTensorTest, SliceNegative2)
{
	Tensor a = Tensor::ones({5, 5, 5});

	auto sliced = a({{1,-3}, {0, -2}, {}}).fill(2.0);

	EXPECT_EQ(sliced.shape, (Shape{1, 3, 5}));
	EXPECT_EQ(sliced.strides, (Strides{100, 20, 4}));

	auto twos = Tensor::empty_like(sliced).fill(2.0);
	EXPECT_TENSOR_EQ(sliced, twos);
}

TEST(CPUTensorTest, SliceNegativeStep)
{
	Tensor a = Tensor::ones({5, 5, 5});
	EXPECT_EQ(a->size, 5*5*5*sizeof(float));

	auto sliced = a({{}, {}, {0, -1,-2}}).fill(2.0);

	EXPECT_EQ(sliced.shape, (Shape{5, 5, 3}));
	EXPECT_EQ(sliced.strides, (Strides{100, 20, -8}));

	EXPECT_EQ(a->size, 5*5*5*sizeof(float));
	EXPECT_EQ(sliced->size, 5*5*5*sizeof(float));

	//std::cout << sliced << std::endl;

	auto twos = Tensor::empty_like(sliced).fill(2.0);

	//std::cout << twos << std::endl;

	EXPECT_TENSOR_EQ(sliced, twos);
}

TEST(CPUTensorTest, SliceStep)
{
	Tensor a = Tensor::ones({5, 5, 5});

	auto sliced = a({{}, {}, {0, -1, 2}}).fill(2.0);

	EXPECT_EQ(sliced.shape, (Shape{5, 5, 3}));
	EXPECT_EQ(sliced.strides, (Strides{100, 20, 8}));

	auto twos = Tensor::empty_like(sliced).fill(2.0);
	EXPECT_TENSOR_EQ(sliced, twos);
}

#define CommonTest TensorTest
#define TestDevice ""
#include "../../Common.hpp"
#include "../../CommonCV.hpp"
#include "../../CommonNN.hpp"
#include "../../CommonLinalg.hpp"
