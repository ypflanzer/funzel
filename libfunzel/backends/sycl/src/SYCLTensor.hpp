#pragma once

#include <funzel/Tensor.hpp>

namespace funzel
{
namespace sycl
{

class SYCLTensor : public BackendTensor
{
public:
	SYCLTensor();
	SYCLTensor(const std::string& args);
	
	const char* backendName() const override { return "SYCL"; }

	void fill(double scalar, DTYPE dtype = FLOAT32) override;
	void empty(std::shared_ptr<char> buffer, const Shape& shape, DTYPE dtype = FLOAT32) override;
	void empty(const void* buffer, const Shape& shape, DTYPE dtype = FLOAT32) override;

	void* data(size_t offset = 0) override;
	std::shared_ptr<char> buffer() override;
	std::shared_ptr<BackendTensor> clone() const override;

	static void initializeBackend();

private:
	std::string m_clArgs;
};

}
}
