#include <gtest/gtest.h>
#include <funzel/nn/Graph.hpp>
#include <funzel/nn/Linear.hpp>

using namespace funzel;

TEST(Graph, Load)
{
	nn::Graph graph;
	nn::GraphNode<nn::Linear> linNode({2, 2, true});
	linNode.add<nn::Linear>(2ULL, 2ULL);
	linNode.add<nn::Linear>(2ULL, 2ULL);
	linNode.add<nn::Linear>(2ULL, 2ULL);
	
	auto nnode = linNode.add<nn::Linear>(2ULL, 2ULL);
	nnode->add<nn::Linear>(2ULL, 2ULL);
	auto nnode2 = nnode->add<nn::Linear>(2ULL, 2ULL);
	linNode.add(nnode2);

	linNode.dump(std::cout);

	auto intensor = Tensor::ones({1, 2, 2});
	linNode.forward(intensor);
}
