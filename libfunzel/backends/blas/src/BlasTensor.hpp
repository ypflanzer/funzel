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

namespace funzel
{
namespace blas
{

class BlasTensor : public BackendTensor
{
public:
	BlasTensor() = default;
	BlasTensor(const std::string&) {}
	
	void fill(const Tensor& self, double scalar) override;
	void empty(std::shared_ptr<char> buffer, size_t sz, const Shape& shape, DTYPE dtype = FLOAT32) override;
	void empty(const void* buffer, size_t sz, const Shape& shape, DTYPE dtype = FLOAT32) override;
	void* data(size_t offset = 0) override;
	std::shared_ptr<char> buffer() override { return m_data; }
	std::shared_ptr<BackendTensor> clone() const override;

	const char* backendName() const override { return "BLAS"; }

	// Operations
	void matmul(const Tensor& self, Tensor b, Tensor tgt) override;
	void mulAdd(const Tensor& self, Tensor tgt, double alpha) override;
	void sub(const Tensor& self, const Tensor& b, double alpha = 1.0) override;

	void abs(const Tensor& self, Tensor tgt) override;
	void exp(const Tensor& self, Tensor tgt) override;
	void sqrt(const Tensor& self, Tensor tgt) override;
	void sin(const Tensor& self, Tensor tgt) override;
	void cos(const Tensor& self, Tensor tgt) override;
	void tan(const Tensor& self, Tensor tgt) override;
	void tanh(const Tensor& self, Tensor tgt) override;
	double sum(const Tensor& self) override;

	void set(Tensor& self, const Tensor& src) override;

	static void initializeBackend();

private:
	std::shared_ptr<char> m_data;
};

}
}
