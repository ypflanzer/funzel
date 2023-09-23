#pragma once

#include "HIPTensor.hip.hpp"
#include "hip/hip_runtime.h"

namespace funzel
{
namespace hip
{

template<typename T>
class EXPORT HIPTensorImpl: public HIPTensor
{
	int m_deviceIdx = -1;
public:
	HIPTensorImpl(int device):
		m_deviceIdx(device)
	{
		dtype = funzel::dtype<T>();
	}

	void empty(const void* data, size_t count) override
	{
		CheckError(hipSetDevice(m_deviceIdx));
		CheckError(hipMalloc(&m_deviceMemory, count*sizeof(T)));
		this->size = count;

		if(data)
		{
			CheckError(hipMemcpy(m_deviceMemory, data, count*sizeof(T), hipMemcpyKind::hipMemcpyHostToDevice));
		}
	}

	template <typename Q>
	static __global__ void memset(Q* dest, size_t count, Q val)
	{
		const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		if(idx < count) dest[idx] = val;
	}

	void fill(const Tensor& self, double scalar) override
	{
		T* dest = (T*)(m_deviceMemory) + (self.offset/sizeof(T));
		if constexpr (std::is_same_v<T, int> || std::is_same_v<T, unsigned int>)
		{
			const int value = scalar;
			CheckError(hipMemset(dest, value, size*sizeof(T)));
		}
		else
		{
			int blockSize = 256;
			int numBlocks = (size + blockSize - 1) / blockSize;
			memset<<<numBlocks, blockSize>>>(dest, size, T(scalar));
		}
	}
};

}
}
