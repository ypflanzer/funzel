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

#include <funzel/Vector.hpp>
#include <cassert>

namespace funzel
{

template<typename T>
void Conv2DNaive(
	const T* input, const T* kernel, T* output,
	const UVec2& inputSize,
	const UVec3& kernelSize,
	const UVec2& outputSize,
	const UVec2& inputStride,
	const UVec2& outputStride,

	const UVec2& stride,
	const UVec2& padding,
	const UVec2& dilation)
{

	const int channels = kernelSize[2] ? kernelSize[2] : 1;
	const int64_t outstrideY = outputStride[0]/sizeof(T) * channels;
	const int64_t outstrideX = outputStride[1]/sizeof(T) * channels;
	const int64_t instrideY = inputStride[0]/sizeof(T);
	const int64_t instrideX = inputStride[1]/sizeof(T);

	const int halfKw = int(kernelSize[1]/2);
	const int halfKh = int(kernelSize[0]/2);

	const int64_t inSizeMax = int64_t(inputSize[0])*inputSize[1];
	const int ksize = kernelSize[0]*kernelSize[1];

	//#pragma omp parallel for
	for(int64_t y = 0; y < outputSize[0]; y++)
	{
		const int64_t inY = y*stride[0];
		const int64_t yinOff = inY*instrideY;

		const int64_t youtOff = y*outstrideY;
		for(int64_t x = 0; x < outputSize[1]; x++)
		{
			const int64_t inX = x*stride[1];
			const int64_t xinOff = inX*instrideX;

			const int64_t xoutOff = x*outstrideX;
			auto* accum = &output[youtOff + xoutOff];

			for (int c = 0; c < channels; c++)
			{
				accum[c] = 0;
			}

			// TODO Optimize!
			for(int64_t ky = -halfKh; ky <= halfKh; ky++)
			{
				for(int64_t kx = -halfKw; kx <= halfKw; kx++)
				{
					const int dkx = dilation[1] * kx + inX;
					const int dky = dilation[0] * ky + inY;

					if (dky < 0 || dky >= inputSize[0]
						|| dkx < 0 || dkx >= inputSize[1])
					{
						continue;
					}

					//std::cout << inY << "*" << instrideY << " + " << ky << "*" << inputSize[1] << " = " << (inY * instrideY + ky * inputSize[1]) << "\t" << (dkx * instrideX) + (inY * instrideY + ky * inputSize[1]) << std::endl;

					const int64_t inputOffset = /*yinOff* + xinOff + */(dkx * instrideX) + (inY * instrideY + ky*inputSize[0]);
					AssertExcept(inputOffset >= 0 && inputOffset < inSizeMax, "");

					//*accum += input[inputOffset] / 25.0f;
					//continue;

					const auto kidx = ((halfKh+ky)*kernelSize[1] + halfKw + kx) * channels;
					const auto inputValue = input[inputOffset];
					for (int c = 0; c < channels; c++)
					{
						AssertExcept(kidx >= 0 && kidx + c < ksize* channels, "");
						assert(kidx >= 0 && kidx + c < ksize*channels);

						accum[c] += inputValue * kernel[kidx + c];
					}
				}
			}
		}
	}
}

}
