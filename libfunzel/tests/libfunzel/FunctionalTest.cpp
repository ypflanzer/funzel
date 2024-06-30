#include <gtest/gtest.h>
#include <funzel/Tensor.hpp>

#include "../TestUtils.hpp"

using namespace funzel;

TEST(Functional, ReduceEmptyAxis)
{
	Tensor t = Tensor::ones({3, 2, 1});
	Tensor out;
	Reduce(t, {}, DTYPE::NONE, out, false, std::optional<int>{0}, [](const Tensor& t1, Tensor& t2) {
		t2.add_(t1);
	});

	const Tensor result = Tensor({1}, {6.0f});
	EXPECT_TENSOR_EQ(result, out);
}

TEST(Functional, ReduceAxis0)
{
	Tensor t = Tensor::ones({3, 2, 1});
	Tensor out;
	Reduce(t, {0}, DTYPE::NONE, out, false, std::optional<int>{0}, [](const Tensor& t1, Tensor& t2) {
		t2.add_(t1);
	});

	const Tensor result = Tensor({2, 1}, {3.0f, 3.0f});
	EXPECT_TENSOR_EQ(result, out);
}

TEST(Functional, ReduceAxis1)
{
	Tensor t = Tensor::ones({3, 2, 1});
	Tensor out;
	Reduce(t, {1}, DTYPE::NONE, out, false, std::optional<int>{0}, [](const Tensor& t1, Tensor& t2) {
		std::cout << t1.shape << " " << t2.shape << std::endl;
		t2.add_(t1);
	});

	const Tensor result = Tensor({3, 1}, {2.0f, 2.0f, 2.0f});
	EXPECT_TENSOR_EQ(result, out);
}

TEST(Functional, ReduceAxis2)
{
	Tensor t = Tensor::ones({3, 2, 1});
	Tensor out;
	Reduce(t, {2}, DTYPE::NONE, out, false, std::optional<int>{0}, [](const Tensor& t1, Tensor& t2) {
		t2.add_(t1);
	});

	const Tensor result = Tensor({3, 2}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,});
	EXPECT_TENSOR_EQ(result, out);
}

TEST(Functional, ReduceAxis12)
{
	Tensor t = Tensor::ones({3, 2, 1});
	Tensor out;
	Reduce(t, {1, 2}, DTYPE::NONE, out, false, std::optional<int>{0}, [](const Tensor& t1, Tensor& t2) {
		t2.add_(t1);
	});

	const Tensor result = Tensor({3}, {2.0f, 2.0f, 2.0f});
	EXPECT_TENSOR_EQ(result, out);
}

TEST(Functional, ReduceAxis21)
{
	Tensor t = Tensor::ones({3, 2, 1});
	Tensor out;
	Reduce(t, {2, 1}, DTYPE::NONE, out, false, std::optional<int>{0}, [](const Tensor& t1, Tensor& t2) {
		t2.add_(t1);
	});

	const Tensor result = Tensor({3}, {2.0f, 2.0f, 2.0f});
	EXPECT_TENSOR_EQ(result, out);
}

TEST(Functional, ReduceAxis02)
{
	Tensor t = Tensor::ones({3, 2, 1});
	Tensor out;
	Reduce(t, {0, 2}, DTYPE::NONE, out, false, std::optional<int>{0}, [](const Tensor& t1, Tensor& t2) {
		t2.add_(t1);
	});

	const Tensor result = Tensor({2}, {3.0f, 3.0f});
	EXPECT_TENSOR_EQ(result, out);
}

TEST(Functional, ReduceAxis01)
{
	Tensor t = Tensor::ones({3, 2, 1});
	Tensor out;
	Reduce(t, {0, 1}, DTYPE::NONE, out, false, std::optional<int>{0}, [](const Tensor& t1, Tensor& t2) {
		t2.add_(t1);
	});

	const Tensor result = Tensor({1}, {6.0f});
	EXPECT_TENSOR_EQ(result, out);
}

TEST(Functional, ReduceAxis2Keepdims)
{
	Tensor t = Tensor::ones({3, 2, 1});
	Tensor out;
	Reduce(t, {2}, DTYPE::NONE, out, true, std::optional<int>{0}, [](const Tensor& t1, Tensor& t2) {
		t2.add_(t1);
	});

	const Tensor result = Tensor({3, 2, 1}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
	EXPECT_TENSOR_EQ(result, out);
}

TEST(Functional, ReduceAxis21Keepdims)
{
	Tensor t = Tensor::ones({3, 2, 1});
	Tensor out;
	Reduce(t, {2, 1}, DTYPE::NONE, out, true, std::optional<int>{0}, [](const Tensor& t1, Tensor& t2) {
		t2.add_(t1);
	});

	const Tensor result = Tensor({3, 1, 1}, {2.0f, 2.0f, 2.0f});
	EXPECT_TENSOR_EQ(result, out);
}

TEST(Functional, ReduceEmptyAxisKeepdims)
{
	Tensor t = Tensor::ones({3, 2, 1});
	Tensor out;
	Reduce(t, {}, DTYPE::NONE, out, true, std::optional<int>{0}, [](const Tensor& t1, Tensor& t2) {
		t2.add_(t1);
	});

	const Tensor result = Tensor({1, 1, 1}, {6.0f});
	EXPECT_TENSOR_EQ(result, out);
}

TEST(Functional, ReduceOutOfRangeAxis)
{
	Tensor t = Tensor::ones({3, 2, 1});
	Tensor out;
	
	EXPECT_THROW(
		Reduce(t, {4}, DTYPE::NONE, out, true, std::optional<int>{0}, [](const Tensor& t1, Tensor& t2) {
			t2.add_(t1);
		}), std::out_of_range
	);
}
