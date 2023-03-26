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

namespace funzel::linalg
{
/**
 * @brief Defines the interface for backend specific computer vision functionality.
 */
class LinalgBackendTensor
{
public:
	virtual ~LinalgBackendTensor() = default;

	static const char* BackendName() { return "Linalg"; }

	virtual void det(const Tensor& self, Tensor tgt) { UnsupportedOperationError; }
	virtual void inv(const Tensor& self, Tensor tgt) { UnsupportedOperationError; }
	virtual void trace(const Tensor& self, Tensor tgt) { UnsupportedOperationError; }
};

}
