#pragma once

#include <funzel/Tensor.hpp>
#include "hip/hip_runtime.h"

namespace funzel
{
namespace hip
{

class HIPTensor : public BackendTensor
{
public:
	static std::shared_ptr<BackendTensor> Empty(std::shared_ptr<char> data, size_t sz, DTYPE dtype, const std::string& args);
	static std::shared_ptr<BackendTensor> Empty(const void* data, size_t sz, DTYPE dtype, const std::string& args);

	HIPTensor();
	HIPTensor(const std::string& args);
	
	const char* backendName() const override { return "HIP"; }

	void* data(size_t offset = 0) override;
	std::shared_ptr<char> buffer() override;
	std::shared_ptr<BackendTensor> clone() const override;

	static void initializeBackend();

	virtual void empty(const void* data, size_t count) = 0;

protected:
	static inline void CheckError(hipError_t err)
	{
		if(err != hipSuccess)
			throw std::runtime_error(std::string("HIP Error: ") + hipGetErrorString(err));
	}

	int m_deviceIdx = -1;
	void* m_deviceMemory = nullptr;

private:
	std::string m_clArgs;
};

}
}
