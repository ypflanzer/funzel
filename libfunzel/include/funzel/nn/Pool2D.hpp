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

#include "Module.hpp"
#include "../Vector.hpp"

namespace funzel
{
namespace nn
{

class FUNZEL_API Pool2D : public Module
{
public:
	Pool2D(const UVec2& kernelSize,
			POOLING_MODE mode,
			const UVec2& stride,
			const UVec2& padding,
			const UVec2& dilation);

	Pool2D(const UVec2& kernelSize, POOLING_MODE mode = MAX_POOLING);

	Tensor forward(const Tensor& input) override;
	Tensor backward(const Tensor& input) override;

	Tensor& weights() { return m_parameters.at(0); }

private:
	POOLING_MODE m_mode = MAX_POOLING;
	UVec2 m_stride = UVec2(1, 1), m_padding = UVec2(0, 0);
	UVec2 m_dilation = UVec2(1, 1), m_kernel = UVec2(1, 1);
};

}
}
