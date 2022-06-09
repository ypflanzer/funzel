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

namespace funzel
{
namespace nn
{

class FUNZEL_API Sequential : public Module
{
public:
	Sequential() = default;
	Sequential(std::initializer_list<ModuleRef> modules);

	Tensor forward(const Tensor& input) final override;
	Tensor backward(const Tensor& input) final override;
	void to(const std::string& device = EmptyStr) final override;
	void defaultInitialize() final override;

private:
	std::vector<ModuleRef> m_modules;
};

}
}
