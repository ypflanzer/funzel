#include <funzel/cv/CV.hpp>
#include <funzel/cv/CVBackendTensor.hpp>

using namespace funzel;
using namespace cv;

Tensor& funzel::cv::conv2d(
	Tensor input,
	Tensor& result,
	const Tensor& kernel,
	const UVec2& stride,
	const UVec2& padding,
	const UVec2& dilation)
{
	input.getBackendAs<cv::CVBackendTensor>()->conv2d(input, result, kernel, stride, padding, dilation);
	return result;
}

Tensor funzel::cv::conv2d(
	Tensor input,
	const Tensor& kernel,
	const UVec2& stride,
	const UVec2& padding,
	const UVec2& dilation)
{
	Tensor result = Tensor::empty_like(input);
	input.getBackendAs<cv::CVBackendTensor>()->conv2d(input, result, kernel, stride, padding, dilation);
	return result;
}
