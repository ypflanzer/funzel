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
#include <funzel/linalg/LinalgBackendTensor.hpp>

namespace funzel
{
namespace blas
{

class EXPORT BlasTensor:
	public BackendTensor,
	public linalg::LinalgBackendTensor,
	public nn::NNBackendTensor,
	public cv::CVBackendTensor
{
public:
	static void initializeBackend();
	static std::shared_ptr<BackendTensor> Empty(std::shared_ptr<char> data, size_t sz, DTYPE dtype, const std::string& args);
	static std::shared_ptr<BackendTensor> Empty(const void* data, size_t sz, DTYPE dtype, const std::string& args);

	BlasTensor() = default;
	BlasTensor(const std::string&) {}

	virtual void empty(std::shared_ptr<char> buffer, size_t sz);
	virtual void empty(const void* buffer, size_t sz);

	void* data(size_t offset = 0) override;
	std::shared_ptr<char> buffer() override { return m_data; }
	std::shared_ptr<BackendTensor> clone() const override;

	const char* backendName() const override { return "BLAS"; }

private:
	std::shared_ptr<char> m_data;
};

}
}
