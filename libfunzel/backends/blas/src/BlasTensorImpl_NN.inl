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

#include "Pool2D.hpp"
#include "Conv2D.hpp"

#include <functional>

namespace funzel
{
namespace blas
{

template<typename T, SIMD_TYPE SimdType>
void BlasTensorImpl<T, SimdType>::pool2d(
			const Tensor& self, Tensor tgt,
			POOLING_MODE mode,
			const UVec2& kernelSize,
			const UVec2& stride,
			const UVec2& padding,
			const UVec2& dilation)
{
	AssertExcept(self.getBackend() != tgt.getBackend(), "Cannot apply a pooling operation in-place!");
	if(self.shape.empty())
		return;
	
	if(self.shape.size() > 2)
	{
		//#pragma omp parallel for
		for(int i = 0; i < self.shape[0]; i++)
		{
			pool2d(self[i], tgt[i], mode, kernelSize, stride, padding, dilation);
		}

		return;
	}

	const void* adata = self.data(self.offset);
	void* dest = tgt.data(tgt.offset);

	const auto maxFunctor = [](const auto a, const auto b, int ksize) { return std::max(a,b); };
	const auto meanFunctor = [](const auto a, const auto b, int ksize) { return a + b/ksize; };

	Pool2D<T>(  (const T*) adata, (T*) dest,
				{self.shape[0], self.shape[1]},
				{tgt.shape[0], tgt.shape[1]},
				{self.strides[0], self.strides[1]},
				{tgt.strides[0], tgt.strides[1]},
				kernelSize, stride, padding, dilation,
				(mode == MEAN_POOLING ?
					std::function<T(const T, const T, int)>(meanFunctor):
					std::function<T(const T, const T, int)>(maxFunctor))
			);
}

template<typename T, SIMD_TYPE SimdType>
void BlasTensorImpl<T, SimdType>::relu(const Tensor& self, Tensor& tgt, double negativeSlope)
{
	if(negativeSlope == 0.0)
		TensorOp(self, tgt, [](const auto& v) {
			using V = typename std::remove_reference<decltype(v)>::type;
			return std::max(V(0), v);
		});
	else
		TensorOp(self, tgt, [negativeSlope](const auto& v) { return v >= 0 ? v : v*negativeSlope; });
}

}
}
