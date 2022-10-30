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

namespace funzel::nn
{
class FUNZEL_API NNBackendTensor
{
public:
	NNBackendTensor() = default;
	virtual ~NNBackendTensor() = default;

	virtual void pool2d(
			const Tensor& self, Tensor tgt,
			POOLING_MODE mode,
			const UVec2& kernelSize,
			const UVec2& stride,
			const UVec2& padding,
			const UVec2& dilation) { ThrowError("Operation is not supported!"); }

	virtual void relu(const Tensor& self, Tensor& tgt, double negativeSlope = 0.0)  { ThrowError("Operation is not supported!"); }

	// With default implementation
	virtual void sigmoid(const Tensor& self, Tensor& tgt);
};

}
