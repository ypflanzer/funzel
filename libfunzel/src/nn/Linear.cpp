#include <funzel/nn/Linear.hpp>

using namespace funzel;
using namespace nn;

Linear::Linear(size_t in, size_t out, bool bias)
{
	m_parameters.reserve(2);
	m_parameters.push_back(Tensor::empty({in, out}));
	
	if(bias)
		m_parameters.push_back(Tensor::empty({out}));
}

Tensor Linear::forward(const Tensor& input)
{
	auto result = input.matmul(weights());
	if(m_parameters.size() > 1)
		result.add_(bias());

	return result;
}

Tensor Linear::backward(const Tensor& input)
{
	return Tensor();
}
