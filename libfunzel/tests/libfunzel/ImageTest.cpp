#include <gtest/gtest.h>
#include <funzel/Image.hpp>
#include <funzel/Plot.hpp>

using namespace funzel;

TEST(Image, LoadInvalid)
{
	EXPECT_THROW(image::load("testnonexist.png"), std::runtime_error);
}

TEST(Image, LoadSave)
{
	auto img = image::load("test.jpg");
	EXPECT_EQ(img.dtype, UBYTE);
	EXPECT_EQ(img.shape, (Shape{1791, 2652, 3}));
	
	img = img.astype<float>();
	EXPECT_EQ(img.dtype, FLOAT32);

	img.mul_(0.5);

	img = img.astype<uint8_t>();
	image::save(img, "test.png");
}

TEST(Image, PlotImage)
{
	auto img = image::load("test.jpg");
	EXPECT_EQ(img.dtype, UBYTE);
	EXPECT_EQ(img.shape, (Shape{1791, 2652, 3}));
	
	Plot plt;
	plt.image(img);
	plt.show();
}
