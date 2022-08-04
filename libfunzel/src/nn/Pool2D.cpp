#include <funzel/nn/Pool2D.hpp>

using namespace funzel;
using namespace nn;

Pool2D::Pool2D(const UVec2& kernelSize, POOLING_MODE mode):
	Pool2D(kernelSize, mode, {1, 1}, {0, 0}, {1, 1}) {}

Pool2D::Pool2D(const UVec2& kernelSize,
				POOLING_MODE mode,
				const UVec2& stride,
				const UVec2& padding,
				const UVec2& dilation)
{
	m_stride = stride;
	m_padding = padding;
	m_kernel = kernelSize;
	m_dilation = dilation;
	m_mode = mode;
}

Tensor Pool2D::forward(const Tensor& input)
{
	Tensor result;
	input->pool2d(input, result, m_mode, m_kernel, m_stride, m_padding, m_dilation);
	return result;
}

Tensor Pool2D::backward(const Tensor& input)
{
	return Tensor();
}
