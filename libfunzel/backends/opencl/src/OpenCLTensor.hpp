/* 
 * This file is part of Funzel.
 * Copyright (c) 2022 Yannick Pflanzer.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
#pragma once

#include "funzel/OpenCLBackend.hpp"
#include <funzel/Tensor.hpp>
#include <funzel/nn/NNBackendTensor.hpp>
#include <funzel/cv/CVBackendTensor.hpp>

namespace funzel
{
namespace cl
{

class CLTemplateKernel;
class OpenCLTensor :
	public BackendTensor,
	public nn::NNBackendTensor,
	public cv::CVBackendTensor
{
public:

	static std::shared_ptr<BackendTensor> Empty(std::shared_ptr<char> data, size_t sz, DTYPE dtype, const std::string& args);
	static std::shared_ptr<BackendTensor> Empty(const void* data, size_t sz, DTYPE dtype, const std::string& args);

	OpenCLTensor();
	OpenCLTensor(const std::string& args);

	void fill(const Tensor& self, double scalar) override;
	void empty(std::shared_ptr<char> buffer, size_t sz, DTYPE dtype);
	void empty(const void* buffer, size_t sz, DTYPE dtype);
	void* data(size_t offset = 0) override;
	std::shared_ptr<char> buffer() override;
	std::shared_ptr<BackendTensor> clone() const override;

	const char* backendName() const override { return "OCL"; }

	// Operations
	void matmul(const Tensor& self, Tensor b, Tensor tgt) override;
	void mulAdd(const Tensor& self, Tensor tgt, double alpha) override;
	void mul(Tensor self, double alpha) override;
	void sub(const Tensor& self, const Tensor& b, double alpha = 1.0) override;
	void div(const Tensor& self, const Tensor& b, Tensor tgt) override;

	void abs(const Tensor& self, Tensor tgt) override;
	void exp(const Tensor& self, Tensor tgt) override;
	void sqrt(const Tensor& self, Tensor tgt) override;
	void sin(const Tensor& self, Tensor tgt) override;
	void cos(const Tensor& self, Tensor tgt) override;
	void tan(const Tensor& self, Tensor tgt) override;
	void tanh(const Tensor& self, Tensor tgt) override;

	void sum(
		const Tensor& self,
		Tensor& tgt,
		const small_vector<int>& axis,
		DTYPE type,
		bool keepdims) override;

	void relu(const Tensor& self, Tensor& tgt, double negativeSlope = 0.0) override;

	void pool2d(
			const Tensor& self, Tensor tgt,
			POOLING_MODE mode,
			const UVec2& kernelSize,
			const UVec2& stride,
			const UVec2& padding,
			const UVec2& dilation) override;

	void conv2d(
			const Tensor& self, Tensor tgt,
			const Tensor& kernel,
			const UVec2& stride,
			const UVec2& padding,
			const UVec2& dilation) override;

	void unravel(const Tensor& self, Tensor tgt) override;

	::cl::Buffer& clbuffer() { return m_buffer; }
	::cl::CommandQueue& clcmdQueue() { return m_device.queue; }
	::cl::Event& clcurrentEvent() { return m_currentEvent; }

protected:

	size_t workgroupSize() const;

	template<typename Fn>
	::cl::Event DoStrided(const Tensor& self, CLTemplateKernel& kernel, Fn&& fn);

	void wait();

	CLDevice m_device;
	::cl::Buffer m_buffer;
	::cl::Event m_currentEvent;
	std::string m_clArgs;

	friend CLTemplateKernel;
};

}
}
