#include <funzel/nn/Linear.hpp>

using namespace funzel;
using namespace nn;

Linear::Linear(size_t in, size_t out)
{
	m_parameters = Tensor::empty({in, out});
}

Tensor Linear::forward(const Tensor& input)
{
	return input.matmul(m_parameters);
}

Tensor Linear::backward(const Tensor& input)
{
	return Tensor();
}

void Linear::to(const std::string& device)
{
	m_parameters = m_parameters.to(device);
}
