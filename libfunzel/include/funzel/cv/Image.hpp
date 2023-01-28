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
#include "../Vector.hpp"

namespace funzel
{
class FUNZEL_API image
{
public:
	image() = delete;

	/**
	 * @brief Defines different channels orders of color images.
	 */
	enum CHANNEL_ORDER
	{
		HWC, ///< height x width x channels
		CHW ///< channels x height x width
	};

	/**
	 * @brief Loads an image file into a Tensor.
	 * 
	 * A channel order and type conversion and will be performed automatically if
	 * required by the arguments.
	 * 
	 * @param file The path to the image file.
	 * @param order The channel order of the new Tensor.
	 * @param dtype The DTYPE of the new Tensor.
	 * @param device The device on which the new Tensor will be created.
	 * @return Tensor A new Tensor of type "dtype" on device "device" with channel order "order".
	 */
	static Tensor load(const std::string& file, CHANNEL_ORDER order = HWC, DTYPE dtype = NONE, const std::string& device = std::string());

	/**
	 * @brief Save an image to a file.
	 * 
	 * @attention The image Tensor needs a HWC channel order!
	 * 
	 * @param tensor The Tensor containing the image.
	 * @param file The file to save to.
	 */
	static void save(const Tensor& tensor, const std::string& file);

	/**
	 * @brief Converts an image Tensor to the required channel order.
	 * @attention This will always permute the channels, make sure the source channel order is correct!
	 * @param t The image Tensor.
	 * @param order The new channel order.
	 * @return Tensor A new Tensor where the channels have been permuted according the the requested channel order.
	 */
	static inline Tensor toOrder(const Tensor& t, CHANNEL_ORDER order)
	{
		if(order == HWC)
			return t.permute({1, 2, 0});

		return t.permute({2, 0, 1});
	}

	/**
	 * @brief Shows the contents of an image Tensor.
	 * 
	 * Creates a new window showing the given image Tensor.
	 * Requires a HWC Tensor of DTYPE DUBYTE.
	 * 
	 * @param t The image Tensor with order HWC.
	 * @param title The window title to show.
	 * @param waitkey Wait for the window to close before returning.
	 */
	static void imshow(const Tensor& t, const std::string& title = "", bool waitkey = false);

	static void drawCircle(Tensor tgt, const Vec2& pos, float r, float thickness = 5, const Vec3& color = Vec3(255, 255, 255));
	static void drawCircles(Tensor tgt, Tensor circlesXYR, float thickness = 5, const Vec3& color = Vec3(255, 255, 255));

	static Tensor gaussianBlur(Tensor input, unsigned int kernelSize, double sigma);
	static Tensor& gaussianBlur(Tensor input, Tensor& tgt, unsigned int kernelSize, double sigma);

	static Tensor sobelDerivative(Tensor input, bool horizontal);
	static Tensor& sobelDerivative(Tensor input, Tensor& tgt, bool horizontal);
};
}
