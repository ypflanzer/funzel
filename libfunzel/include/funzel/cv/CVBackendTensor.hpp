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
#include <funzel/Vector.hpp>

namespace funzel::cv
{
/**
 * @brief Defines the interface for backend specific computer vision functionality.
 */
class CVBackendTensor
{
public:
	virtual ~CVBackendTensor() = default;

	// Disable unused parameter warnings for the following method stubs
	#if defined(__GNUC__) || defined(__clang__)
		#pragma GCC diagnostic push
		#pragma GCC diagnostic ignored "-Wunused-parameter"
	#elif defined(_MSC_VER)
		#pragma warning(push)
		#pragma warning(disable : 4100) // Disable unused parameter warning
	#endif

	/**
	 * @brief Converts an image Tensor to grayscale.
	 * 
	 * @param self The source Tensor.
	 * @param tgt The target Tensor.
	 */
	virtual void convertGrayscale(const Tensor& self, Tensor tgt) { UnsupportedOperationError; }

	/**
	 * @brief Applies a 2D convolution to an image Tensor.
	 * 
	 * @param self The input image Tensor.
	 * @param tgt The target image Tensor.
	 * @param kernel A convolution kernel of shape (H, W, C).
	 * @param stride The stride of the kernel while moving over the image.
	 * @param padding The padding applied at the edges of the images.
	 * @param dilation The dilation of the individual kernel elements.
	 */
	virtual void conv2d(
			const Tensor& self, Tensor tgt,
			const Tensor& kernel,
			const UVec2& stride,
			const UVec2& padding,
			const UVec2& dilation) { UnsupportedOperationError; }

	#if defined(__GNUC__) || defined(__clang__)
		#pragma GCC diagnostic pop
	#elif defined(_MSC_VER)
		#pragma warning(pop)
	#endif
};

}
