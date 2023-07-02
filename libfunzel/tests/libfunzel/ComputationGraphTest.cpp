#include <gtest/gtest.h>
#include <funzel/nn/Graph.hpp>
#include <funzel/nn/Linear.hpp>

using namespace funzel;
using namespace nn;

TEST(Graph, Load)
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
}

TEST(Graph, Eval)
{

}
