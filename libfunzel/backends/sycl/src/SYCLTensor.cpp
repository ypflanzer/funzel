#include <funzel/Tensor.hpp>
#include "SYCLTensor.hpp"

#include <SYCL/sycl.hpp>

#include <iostream>

using namespace funzel;
using namespace funzel::sycl;

using namespace ::sycl;

void SYCLTensor::initializeBackend()
{
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
}

SYCLTensor::SYCLTensor():
	SYCLTensor("0") {}

SYCLTensor::SYCLTensor(const std::string& args):
	m_clArgs(args)
{
	
}

void SYCLTensor::fill(double scalar, DTYPE dtype)
{

}

void SYCLTensor::empty(std::shared_ptr<char> buffer, const Shape& shape, DTYPE dtype)
{

}

void SYCLTensor::empty(const void* buffer, const Shape& shape, DTYPE dtype)
{

}

void* SYCLTensor::data(size_t offset)
{

}

std::shared_ptr<char> SYCLTensor::buffer()
{

}

std::shared_ptr<BackendTensor> SYCLTensor::clone() const
{

}
