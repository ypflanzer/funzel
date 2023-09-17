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

namespace funzel
{
namespace blas
{

template<typename T>
void BlasTensorImpl<T>::conv2d(
		const Tensor& self, Tensor tgt,
		const Tensor& kernel,
		const UVec2& stride,
		const UVec2& padding,
		const UVec2& dilation)
{
	AssertExcept(self.getBackend() != tgt.getBackend(), "Cannot apply a convolution operation in-place!");
	if (self.shape.empty())
		return;

	if (self.shape.size() > 2)
	{
		//#pragma omp parallel for
		for (int i = 0; i < self.shape[0]; i++)
		{
			conv2d(self[i], tgt[i], kernel[i], stride, padding, dilation);
		}

		return;
	}

	const void* adata = self.data(self.offset);
	void* dest = tgt.data(tgt.offset);
	const void* kdata = kernel.data(kernel.offset);

	const size_t oh = ((self.shape[0] + size_t(2) * padding[0] - dilation[0] * (kernel.shape[0] - 1) - 1) / stride[0]) + 1;
	const size_t ow = ((self.shape[1] + size_t(2) * padding[1] - dilation[1] * (kernel.shape[1] - 1) - 1) / stride[1]) + 1;
	AssertExcept(tgt.shape[0] == oh && tgt.shape[1] == ow, "Invalid output size: " << tgt.shape[0] << " != " << oh << " or " << tgt.shape[1] << " != " << ow);

	Conv2DNaive<T>(
			(const T*) adata, (const T*) kdata, (T*) dest,
			{ self.shape[0], self.shape[1] },
			{ kernel.shape[0], kernel.shape[1] },
			{ tgt.shape[0], tgt.shape[1] },
			{ self.strides[0], self.strides[1] },
			{ tgt.strides[0], tgt.strides[1] },
			stride, padding, dilation);
}

template<typename V>
static void ConvertRGBToGrayCHW(const Tensor& self, Tensor& tgt)
{
	#pragma omp parallel for
	for(int64_t y = 0; y < self.shape[1]; y++)
	{
		const size_t yoffIn = (y*self.strides[1])/sizeof(V);
		const size_t yoffOut = (y*tgt.strides[1])/sizeof(V);
		
		for(int64_t x = 0; x < self.shape[2]; x++)
		{
			const size_t xoffIn = (x*self.strides[2])/sizeof(V);
			const size_t xoffOut = (x*tgt.strides[2])/sizeof(V);

			V accum = V(0);
			for(int c = 0; c < self.shape[0]; c++)
			{
				const size_t inOff = (yoffIn + xoffIn + (c*self.strides[0]/sizeof(V)));
				accum += self.dataAs<V>(inOff);
			}

			const size_t outOff = xoffOut + yoffOut;
			tgt.dataAs<V>(outOff) = accum / self.shape[0];
		}
	}
}

template<typename T>
void BlasTensorImpl<T>::convertGrayscale(const Tensor& self, Tensor tgt)
{
	if(self.shape.size() <= 2)
		return;
	
	if(self.shape.size() > 3)
	{
		//#pragma omp parallel for
		for(int i = 0; i < self.shape[0]; i++)
		{
			convertGrayscale(self[i], tgt[i]);
		}

		return;
	}

	ConvertRGBToGrayCHW<T>(self, tgt);
}

}
}
