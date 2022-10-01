#include <funzel/cv/Image.hpp>
#include <filesystem>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STBI_ASSERT(x) AssertExcept((x), "Assertion failed: " #x)

#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#define STBIW_ASSERT(x) AssertExcept((x), "Assertion failed: " #x)
#include "stb_image_write.h"

using namespace funzel;
using namespace image;

Tensor funzel::image::load(const std::string& file, CHANNEL_ORDER order, DTYPE dtype, const std::string& device)
{
	std::shared_ptr<char> buffer;
	int w, h, c;

	// Determine appropriate dtype
	if(dtype == NONE)
	{
		if(stbi_is_16_bit(file.c_str()))
			dtype = UINT16;
		if(stbi_is_hdr(file.c_str()))
			dtype = FLOAT32;
		else
			dtype = UBYTE;
	}

	switch(dtype)
	{
	case UINT16:
	case INT16:
		buffer = std::shared_ptr<char>((char*) stbi_load_16(file.c_str(), &w, &h, &c, 0));
		break;

	case FLOAT32:
		buffer = std::shared_ptr<char>((char*) stbi_loadf(file.c_str(), &w, &h, &c, 0));
		break;
	
	default:
	case BYTE:
	case UBYTE:
		buffer = std::shared_ptr<char>((char*) stbi_load(file.c_str(), &w, &h, &c, 0));
		break;
	}

	AssertExcept(buffer != nullptr, "Could not load image: " << stbi_failure_reason());

	auto image = Tensor::empty({size_t(h), size_t(w), size_t(c)}, buffer, dtype, device);
	if(order == CHW)
	{
		image = image::toOrder(image, CHW).unravel();
	}

	return image;
}

static int saveUByteImage(const Tensor& tensor, const std::string& file, const std::string& ext)
{
	const auto h = tensor.shape[0];
	const auto w = tensor.shape[1];
	const auto c = (tensor.shape.size() > 2 ? tensor.shape[2] : 1);
	const void* data = tensor.data();

	AssertExcept(c <= 3, "More than three color channels given, maybe the image is in a CHW format instead of HWC?");

	if(ext == "png")
	{
		return stbi_write_png(file.c_str(), w, h, c, data, 0);
	}
	else if(ext == "jpg" || ext == "jpeg")
	{
		return stbi_write_jpg(file.c_str(), w, h, c, data, 90);
	}
	else if(ext == "bmp")
	{
		return stbi_write_bmp(file.c_str(), w, h, c, data);
	}
	else if(ext == "tga")
	{
		return stbi_write_tga(file.c_str(), w, h, c, data);
	}
	
	AssertExcept(false, "Unsupported image format " << ext << " for dtype " << dtypeToNativeString(tensor.dtype));
}

// TODO: Implement saving ushort!
static int saveUShortImage(const Tensor& tensor, const std::string& file, const std::string& ext)
{
	AssertExcept(false, "Unsupported image format " << ext << " for dtype " << dtypeToNativeString(tensor.dtype));
}

static int saveFloatImage(const Tensor& tensor, const std::string& file, const std::string& ext)
{
	const auto h = tensor.shape[0];
	const auto w = tensor.shape[1];
	const auto c = (tensor.shape.size() > 2 ? tensor.shape[2] : 1);
	const float* data = (float*) tensor.data();

	AssertExcept(c <= 3, "More than three color channels given, maybe the image is in a CHW format instead of HWC?");

	if(ext == "hdr")
	{
		return stbi_write_hdr(file.c_str(), w, h, c, data);
	}
	
	AssertExcept(false, "Unsupported image format " << ext << " for dtype " << dtypeToNativeString(tensor.dtype));
}

void funzel::image::save(const Tensor& tensor, const std::string& file)
{
	std::filesystem::path path(file);
	auto ext = path.extension().string().substr(1);
	std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

	Tensor contiguousTensor;
	if(!tensor.isContiguous())
		contiguousTensor = tensor.unravel();
	else
		contiguousTensor = tensor;

	int err = 0;
	switch(contiguousTensor.dtype)
	{
	case UINT16:
	case INT16:
		err = saveUShortImage(contiguousTensor, file, ext);
		break;

	case FLOAT32:
		err = saveFloatImage(contiguousTensor, file, ext);
		break;
	
	default:
	case BYTE:
	case UBYTE:
		err = saveUByteImage(contiguousTensor, file, ext);
		break;
	}

	AssertExcept(err, "Could not write image as " << ext << ": " << stbi_failure_reason());
}

#include <FL/Fl_Image.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Box.H>
#include <FL/Fl.H>

#include <thread>

void image::imshow(const Tensor& t, const std::string& title, bool waitkey)
{
	auto fn = [t, title]() {

		Tensor im;
		if(!t.isContiguous())
			im = t.unravel();
		else
			im = t;

		const int w = im.shape[1];
		const int h = im.shape[0];
		const int c = im.shape[2];
		const int stride = im.strides[0];
		
		Fl_Window win(w, h, title.c_str());
		Fl_RGB_Image img((const uchar*) im.data(), w, h, c, stride);
		Fl_Box box(0, 0, w, h);

		box.image(img);
		win.add(box);

		win.show();
		Fl::run();
	};

	if(waitkey)
	{
		fn();
	}
	else
	{
		std::thread thr(fn);
		thr.detach();
	}
}
