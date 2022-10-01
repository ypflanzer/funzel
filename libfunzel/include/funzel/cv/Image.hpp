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

#include "../Tensor.hpp"

namespace funzel
{
namespace image
{
	enum CHANNEL_ORDER
	{
		HWC,
		CHW
	};

	FUNZEL_API Tensor load(const std::string& file, CHANNEL_ORDER order = HWC, DTYPE dtype = NONE, const std::string& device = std::string());
	FUNZEL_API void save(const Tensor& tensor, const std::string& file);

	inline Tensor toOrder(const Tensor& t, CHANNEL_ORDER order)
	{
		if(order == HWC)
			return t.permute({1, 2, 0});

		return t.permute({2, 0, 1});
	}

	FUNZEL_API void imshow(const Tensor& t, const std::string& title = "", bool waitkey = false);
}
}
