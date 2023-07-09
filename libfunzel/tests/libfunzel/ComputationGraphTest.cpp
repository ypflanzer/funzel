#include <gtest/gtest.h>
#include <funzel/nn/Graph.hpp>
#include <funzel/nn/Linear.hpp>

#include "../TestUtils.hpp"

using namespace funzel;
using namespace nn;

TEST(Graph, SimpleAddition)
{
	auto a = std::make_shared<ConstantNode>();
	auto b = std::make_shared<ConstantNode>();
	auto c = std::make_shared<ConstantNode>();

	a->value() = Tensor::ones({32});
	b->value() = Tensor::ones({32});
	c->value() = Tensor::ones({32});

	auto add = std::make_shared<AddNode>(a->result(), b->result());
	auto add2 = std::make_shared<AddNode>(a->result(), add->result());

	auto result = add2->result();

	// (1 + 1) + 1 = 3
	auto expected = Tensor::ones({32}) * 3;
	EXPECT_TENSOR_EQ(Tensor(result), expected);
}
