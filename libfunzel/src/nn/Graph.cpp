#include <funzel/nn/Graph.hpp>

using namespace funzel;
using namespace nn;

#if 0

Tensor Graph::forward(const Tensor& input)
{
	for(const auto& v : children)
	{
		
	}
}

Tensor Graph::backward(const Tensor& input)
{
	return Tensor();
}

void Graph::to(const std::string& device)
{

}

void Graph::defaultInitialize()
{

}

void Graph::dump(std::ostream& out)
{
	for(const auto& v : children)
	{
		v->dump(out);
	}
}

#endif
