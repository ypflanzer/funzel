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

TEST(Image, DrawCircle)
{
	auto img = image::load("color_image.png");
	EXPECT_EQ(img.dtype, UBYTE);
	EXPECT_EQ(img.shape, (Shape{28, 28, 3}));
	
	image::drawCircle(img, {14, 14}, 10, 4);
	// image::imshow(img, "", true);

	image::save(img, "Image_DrawCircle.png");
}

TEST(Image, DrawCircles)
{
	auto img = image::load("color_image.png");
	EXPECT_EQ(img.dtype, UBYTE);
	EXPECT_EQ(img.shape, (Shape{28, 28, 3}));
	
	Tensor circles({3, 3}, {
		5.0f, 5.0f, 4.0f,
		10.0f, 10.0f, 4.0f,
		20.0f, 20.0f, 5.0f
	});

	image::drawCircles(img, circles, 2);
	// image::imshow(img, "", true);

	image::save(img, "Image_DrawCircles.png");
}
