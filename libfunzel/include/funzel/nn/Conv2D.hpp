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

class FUNZEL_API Conv2D : public Module
{
public:
	Conv2D(
	size_t inChannels, size_t outChannels,
	const UVec2& kernelSize,
	const UVec2& stride,
	const UVec2& padding,
	const UVec2& dilation,
	bool bias = true);

	Conv2D(size_t inChannels, size_t outChannels, const UVec2& kernelSize, bool bias = true);

	Tensor forward(const Tensor& input) override;
	Tensor backward(const Tensor& input) override;

	Tensor& bias() { return m_parameters.at(1); }
	Tensor& weights() { return m_parameters.at(0); }

private:
	UVec2 m_stride = UVec2(1, 1), m_padding = UVec2(0, 0), m_dilation = UVec2(1, 1);
};

}
}
