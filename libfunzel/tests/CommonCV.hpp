// No pragma once, we want it multiple times!

#include <gtest/gtest.h>
#include <funzel/Tensor.hpp>
#include <funzel/nn/NNBackendTensor.hpp>
#include <funzel/cv/CVBackendTensor.hpp>

#include <cmath>

using namespace funzel;

#ifndef CommonTest
#define CommonTest DefaultTest
#endif

#ifndef TestDevice
#define TestDevice ""
#endif

#define CommonTestCV CAT(CommonTest, _CV)

#include <funzel/cv/Image.hpp>
#include <funzel/Plot.hpp>

TEST(CommonTestCV, Conv2d)
{
	auto img = image::load("mnist_png/training/8/225.png").astype<float>().to(TestDevice).mul_(1.0 / 255.0);
	img.shape.erase(img.shape.begin() + 2);
	img.reshape_(img.shape);

	auto tgt = Tensor::zeros_like(img);
	
#if 1
	auto kernel = Tensor::ones({ 5, 5 }, DFLOAT32, TestDevice);
	kernel.mul_(1.0 / (5.0*5.0));
	img.getBackendAs<cv::CVBackendTensor>()->conv2d(img, tgt, kernel, { 1, 1 }, { 2, 2 }, { 1, 1 });
#else
	Tensor kernel({ 5, 5 }, {
		0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, -1.0f, 0.0f,
		0.0f, 1.0f, 0.0f, -1.0f, 0.0f,
		0.0f, 1.0f, 0.0f, -1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
	}, TestDevice);
	kernel.mul_(1.0 / 3.0);
	img.getBackendAs<nn::NNBackendTensor>()->conv2d(img, tgt, kernel, { 1, 1 }, { 2, 2 }, { 1, 1 });
#endif

	//tgt.shape.push_back(1);
	//tgt.permute_({1, 2, 0});

	tgt = tgt.cpu().mul_(255.0).astype<uint8_t>();
	image::save(tgt, STRINGIFY(CommonTestCV) "_Conv2d.png");

#if 1
	Plot plt;
	//plt.image(img.mul(255.0).cpu().astype<uint8_t>());
	plt.image(tgt, "Result");
	plt.show();
#endif
}

TEST(CommonTestCV, Conv2dColor)
{
	auto img = image::load("test.jpg", image::CHW).astype<float>().mul_(1.0 / 255.0).to(TestDevice);
	auto tgt = Tensor::zeros_like(img);
	
	auto kernel = Tensor::ones({ 5, 5, 3 }, DFLOAT32, TestDevice);
	kernel.mul_(1.0 / (5.0*5.0));
	img.getBackendAs<cv::CVBackendTensor>()->conv2d(img, tgt, kernel, { 1, 1 }, { 2, 2 }, { 1, 1 });

	tgt = tgt.cpu().mul_(255.0).astype<uint8_t>();
	tgt = image::toOrder(tgt, image::HWC);
	image::save(tgt, STRINGIFY(CommonTestCV) "_Conv2d.png");

#if 1
	Plot plt;
	//plt.image(img.mul(255.0).cpu().astype<uint8_t>());
	plt.image(tgt, "Result");
	plt.show();
#endif
}

TEST(CommonTestCV, ConvertToGrayscale)
{
	auto img = image::load("test.jpg", image::CHW).astype<float>().to(TestDevice);
	auto tgt = Tensor::empty({1, img.shape[1], img.shape[2]}, img.dtype);

	img.getBackendAs<cv::CVBackendTensor>()->convertGrayscale(img, tgt);

	tgt = tgt.cpu().astype<uint8_t>();
	tgt = image::toOrder(tgt, image::HWC);
	image::save(tgt, STRINGIFY(CommonTestCV) "_Grayscale.png");
	image::imshow(tgt, "Grayscale", true);
}

#undef CommonTestCV
