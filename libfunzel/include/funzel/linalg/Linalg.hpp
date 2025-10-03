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
namespace linalg
{

struct SVDResult
{
	Tensor u, s, vh;
};

FUNZEL_API Tensor& det(Tensor input, Tensor& result);
FUNZEL_API Tensor det(Tensor input);

FUNZEL_API Tensor& inv(Tensor input, Tensor& result);
FUNZEL_API Tensor inv(Tensor input);

FUNZEL_API Tensor& trace(Tensor input, Tensor& result);
FUNZEL_API Tensor trace(Tensor input);

FUNZEL_API void svd(Tensor input, Tensor& U, Tensor& S, Tensor& V, bool fullMatrices = true);
FUNZEL_API SVDResult svd(Tensor input, bool fullMatrices = true);
FUNZEL_API Tensor svdvals(Tensor input);
}
}
