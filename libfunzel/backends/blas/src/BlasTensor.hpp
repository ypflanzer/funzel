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

#include <funzel/Tensor.hpp>
#include <funzel/nn/NNBackendTensor.hpp>
#include <funzel/cv/CVBackendTensor.hpp>

namespace funzel
{
namespace blas
{

class EXPORT BlasTensor:
	public BackendTensor,
	public nn::NNBackendTensor,
	public cv::CVBackendTensor
{
public:
	static void initializeBackend();

	BlasTensor() = default;
	BlasTensor(const std::string&) {}
	
	void fill(const Tensor& self, double scalar) override;
	void empty(std::shared_ptr<char> buffer, size_t sz, DTYPE dtype = DFLOAT32) override;
	void empty(const void* buffer, size_t sz, DTYPE dtype = DFLOAT32) override;
	void* data(size_t offset = 0) override;
	std::shared_ptr<char> buffer() override { return m_data; }
	std::shared_ptr<BackendTensor> clone() const override;

	const char* backendName() const override { return "BLAS"; }

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
	double sum(const Tensor& self) override;

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

	void set(Tensor& self, const Tensor& src) override;
	void unravel(const Tensor& self, Tensor tgt) override;

	// cv backend
	void convertGrayscale(const Tensor& self, Tensor tgt) override;

private:
	std::shared_ptr<char> m_data;
};

}
}
