#include <funzel/Tensor.hpp>
#include "HIPTensor.hip.hpp"

#include "HIPTensorImpl.hip.hpp"

#include <spdlog/spdlog.h>

using namespace funzel;
using namespace funzel::hip;

static inline HIPTensor* CreateBlasTensor(DTYPE dtype, int device)
{
	switch(dtype)
	{
		default:
		case DFLOAT32: return new HIPTensorImpl<float>(device); break;
		case DFLOAT64: return new HIPTensorImpl<double>(device); break;
		case DINT8: return new HIPTensorImpl<int8_t>(device); break;
		case DINT16: return new HIPTensorImpl<int16_t>(device); break;
		case DINT32: return new HIPTensorImpl<int32_t>(device); break;
		case DINT64: return new HIPTensorImpl<int64_t>(device); break;
		case DUINT8: return new HIPTensorImpl<uint8_t>(device); break;
		case DUINT16: return new HIPTensorImpl<uint16_t>(device); break;
		case DUINT32: return new HIPTensorImpl<uint32_t>(device); break;
	}

	// Should never happen!
	return nullptr;
}

std::shared_ptr<BackendTensor> HIPTensor::Empty(std::shared_ptr<char> data, size_t sz, DTYPE dtype, const std::string& args)
{
	return Empty(data.get(), sz, dtype, args);
}

std::shared_ptr<BackendTensor> HIPTensor::Empty(const void* data, size_t sz, DTYPE dtype, const std::string& args)
{
	int device = 0;
	if(!args.empty())
		device = std::stol(args);

	HIPTensor* tensor = CreateBlasTensor(dtype, device);
	tensor->empty(data, sz);
	return std::shared_ptr<BackendTensor>(tensor);
}

void HIPTensor::initializeBackend()
{
	int numDevices = 0;
	
	if(hipGetDeviceCount(&numDevices) != hipSuccess || numDevices == 0)
	{
		spdlog::debug("No HIP devices found.");
		return;
	}

	spdlog::debug("Found {} HIP device(s).", numDevices);

	for(int i = 0; i < numDevices; i++)
	{
		//CheckError(hipSetDevice(i));
		hipDevice_t device;
		CheckError(hipDeviceGet(&device, i));

		DeviceProperties props;
		props.deviceID = "HIP:" + std::to_string(i);
		props.vendorName = "AMD HIP";
		props.isGPU = true;

		char buf[256];
		CheckError(hipDeviceGetName(buf, sizeof(buf), device));
		props.deviceName = buf;
		
		CheckError(hipDeviceTotalMem(&props.memorySize, device));

		spdlog::debug("\t{}\t{}\t{}", props.deviceID, props.vendorName, props.deviceName);
		backend::RegisterDevice(props);
	}
}

HIPTensor::HIPTensor():
	HIPTensor("0") {}

HIPTensor::HIPTensor(const std::string& args):
	m_clArgs(args)
{
	
}

void* HIPTensor::data(size_t offset)
{
	return nullptr;
}

std::shared_ptr<char> HIPTensor::buffer()
{
	const size_t sz = this->size*dtypeSizeof(dtype);
	char* buffer = new char[sz];

	CheckError(hipMemcpy(buffer, m_deviceMemory, sz, hipMemcpyKind::hipMemcpyDeviceToHost));
	return std::shared_ptr<char>(buffer);
}

std::shared_ptr<BackendTensor> HIPTensor::clone() const
{
	return nullptr;
}
