/* 
 * This file is part of Funzel.
 * Copyright (c) 2022-2023 Yannick Pflanzer.
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
#include "Image.hpp"

namespace funzel::cv
{
FUNZEL_API Tensor& conv2d(
	Tensor input,
	Tensor& result,
	const Tensor& kernel,
	const UVec2& stride,
	const UVec2& padding,
	const UVec2& dilation);

FUNZEL_API Tensor conv2d(
	Tensor input,
	const Tensor& kernel,
	const UVec2& stride,
	const UVec2& padding,
	const UVec2& dilation);

FUNZEL_API Tensor grayscale(Tensor input, image::CHANNEL_ORDER order = image::HWC);
}
