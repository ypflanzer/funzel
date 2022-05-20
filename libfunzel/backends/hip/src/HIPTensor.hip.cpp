#include <funzel/Tensor.hpp>
#include "HIPTensor.hip.hpp"

#include <iostream>

#include "hip/hip_runtime.h"

__device__ void MyKernel()
{
	float f = 4;
}

using namespace funzel;
using namespace funzel::hip;

void HIPTensor::initializeBackend()
{
	#if 0
	auto devices = device::get_devices(info::device_type::all);

	int ctr = 0;
	for(auto& d : devices)
	{
		DeviceProperties props;
		props.deviceID = "SYCL:" + std::to_string(ctr++);
		props.deviceName = d.get_info<info::device::name>();
		props.vendorName = d.get_info<info::device::vendor>();
		props.memorySize = d.get_info<info::device::global_mem_size>();
		props.isGPU = d.is_gpu();

		backend::RegisterDevice(props);
	}
	#endif
}

HIPTensor::HIPTensor():
	HIPTensor("0") {}

HIPTensor::HIPTensor(const std::string& args):
	m_clArgs(args)
{
	
}

void HIPTensor::fill(const Tensor& self, double scalar)
{

}

void HIPTensor::empty(std::shared_ptr<char> buffer, const Shape& shape, DTYPE dtype)
{

}

void HIPTensor::empty(const void* buffer, const Shape& shape, DTYPE dtype)
{

}

void* HIPTensor::data(size_t offset)
{

}

std::shared_ptr<char> HIPTensor::buffer()
{

}

std::shared_ptr<BackendTensor> HIPTensor::clone() const
{

}
