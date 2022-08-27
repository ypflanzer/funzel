#include <funzel/nn/NNBackendTensor.hpp>

using namespace funzel;
using namespace nn;

void NNBackendTensor::sigmoid(const Tensor& self, Tensor& tgt)
{
	tgt = (1.0 / (1.0 + (-self).exp_()));
}
