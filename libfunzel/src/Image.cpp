#include "funzel/Image.hpp"
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

Tensor funzel::image::load(const std::string& file, DTYPE dtype, const std::string& device)
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
	return Tensor::empty({size_t(w), size_t(h), size_t(c)}, buffer, dtype, device);
}


static int saveUByteImage(const Tensor& tensor, const std::string& file, const std::string& ext)
{
	const auto w = tensor.shape[0];
	const auto h = tensor.shape[1];
	const auto c = tensor.shape[2];
	const void* data = tensor.data();

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
	const auto w = tensor.shape[0];
	const auto h = tensor.shape[1];
	const auto c = tensor.shape[2];
	const float* data = (float*) tensor.data();

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

	int err = 0;
	switch(tensor.dtype)
	{
	case UINT16:
	case INT16:
		err = saveUShortImage(tensor, file, ext);
		break;

	case FLOAT32:
		err = saveFloatImage(tensor, file, ext);
		break;
	
	default:
	case BYTE:
	case UBYTE:
		err = saveUByteImage(tensor, file, ext);
		break;
	}

	AssertExcept(err, "Could not write image as " << ext << ": " << stbi_failure_reason());
}
