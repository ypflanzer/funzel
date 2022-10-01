#include <gtest/gtest.h>
#include <funzel/cv/Image.hpp>
#include <funzel/Plot.hpp>

using namespace funzel;

TEST(Image, LoadInvalid)
{
	EXPECT_THROW(image::load("testnonexist.png"), std::runtime_error);
}

TEST(Image, LoadSave)
{
	auto img = image::load("mnist_png/training/8/225.png");
	EXPECT_EQ(img.dtype, UBYTE);
	EXPECT_EQ(img.shape, (Shape{28, 28, 1}));
	
	img = img.astype<float>();
	EXPECT_EQ(img.dtype, FLOAT32);

	img.mul_(0.5);

	img = img.astype<uint8_t>();
	image::save(img, "test.png");
}

TEST(Image, PlotImage)
{
	auto img = image::load("mnist_png/training/8/225.png");
	EXPECT_EQ(img.dtype, UBYTE);
	EXPECT_EQ(img.shape, (Shape{28, 28, 1}));
	
	Plot plt;
	plt.image(img);
	plt.save("Image_PlotImage.png");
}

TEST(Image, PlotImageHWC)
{
	auto img = image::load("color_image.png");
	EXPECT_EQ(img.dtype, UBYTE);
	EXPECT_EQ(img.shape, (Shape{28, 28, 3}));
	
	image::save(img, "Image_ImageHWC.png");
}

TEST(Image, PlotImageCHW)
{
	auto img = image::load("color_image.png", image::CHW);
	EXPECT_EQ(img.dtype, UBYTE);
	EXPECT_EQ(img.shape, (Shape{3, 28, 28}));

	image::save(image::toOrder(img, image::HWC), "Image_ImageCHW.png");
}
