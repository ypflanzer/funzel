#include <funzel/nn/Conv2D.hpp>
#include <funzel/nn/NNBackendTensor.hpp>

using namespace funzel;
using namespace nn;

Conv2D::Conv2D(size_t inChannels, size_t outChannels, const UVec2& kernelSize, bool bias)
{
	m_parameters.reserve(2);
	m_parameters.push_back(Tensor::empty({kernelSize[0], kernelSize[1], outChannels}));
	
	if(bias)
		m_parameters.push_back(Tensor::empty({outChannels}));
}

Conv2D::Conv2D(size_t inChannels, size_t outChannels,
				const UVec2& kernelSize,
				const UVec2& stride,
				const UVec2& padding,
				const UVec2& dilation,
				bool bias): Conv2D(inChannels, outChannels, kernelSize, bias)
{
	m_stride = stride;
	m_padding = padding;
	m_dilation = dilation;
}

Tensor Conv2D::forward(const Tensor& input)
{
	Tensor result;
	input.getBackendAs<nn::NNBackendTensor>()->conv2d(input, result, weights(), m_stride, m_padding, m_dilation);
	
	if(m_parameters.size() > 1)
		result.add_(bias());

	return result;
}

Tensor Conv2D::backward(const Tensor& input)
{
	return Tensor();
}
