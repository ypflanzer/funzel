#include <funzel/cv/Image.hpp>
#include <filesystem>
#include <algorithm>

#include <funzel/cv/CVBackendTensor.hpp>
#include <funzel/nn/NNBackendTensor.hpp>

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STBI_ASSERT(x) AssertExcept((x), "Assertion failed: " #x)

#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#define STBIW_ASSERT(x) AssertExcept((x), "Assertion failed: " #x)
#include "stb_image_write.h"

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

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

		const uint w = im.shape[1];
		const uint h = im.shape[0];
		const uint c = im.shape[2];
		const uint stride = im.strides[0];
		
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

inline void PutPixel(Tensor& tgt, uint x, uint y, const Vec3& color)
{
	auto px = tgt[{x, y}];
	for(int c = 0; c < tgt.shape.back(); c++)
		px[c] = color[c];
}

static void DrawCircleInternal(Tensor& tgt, uint xc, uint yc, uint x, uint y, const Vec3& color)
{
	PutPixel(tgt, xc+x, yc+y, color);
	PutPixel(tgt, xc-x, yc+y, color);
	PutPixel(tgt, xc+x, yc-y, color);
	PutPixel(tgt, xc-x, yc-y, color);
	PutPixel(tgt, xc+y, yc+x, color);
	PutPixel(tgt, xc-y, yc+x, color);
	PutPixel(tgt, xc+y, yc-x, color);
	PutPixel(tgt, xc-y, yc-x, color);
}

// Bresenham!
static void DrawCircleInternal(Tensor tgt, const Vec2& pos, float r, const Vec3& color)
{
	int64_t x = 0, y = r;
	int64_t d = 3 - 2 * r;

	DrawCircleInternal(tgt, pos[0], pos[1], x, y, color);

	while (y >= x)
	{
		x++;

		if (d > 0)
		{
			y--;
			d = d + 4 * (x - y) + 10;
		}
		else
			d = d + 4 * x + 6;

		DrawCircleInternal(tgt, pos[0], pos[1], x, y, color);
	}
}

void image::drawCircle(Tensor tgt, const Vec2& pos, float r, float thickness, const Vec3& color)
{
	AssertExcept(tgt.isContiguous() && tgt.shape.size() == 3 && tgt.shape.back() <= 3,
					"The target tensor needs to be contiguous and formatted as HWC!");

	// Draw multiple circles to reach required thickness
	for(int ir = r - thickness/2; ir <= r + thickness/2; ir++)
	{
		DrawCircleInternal(tgt, pos, ir, color);
	}
}

void image::drawCircles(Tensor tgt, Tensor circlesXYR, float thickness, const Vec3& color)
{
	AssertExcept(tgt.isContiguous() && tgt.shape.size() == 3 && tgt.shape.back() <= 3,
					"The target tensor needs to be contiguous and formatted as HWC!");
	AssertExcept(circlesXYR.shape.size() == 2 && tgt.shape.back() == 3,
					"The circle tensor needs to be of shape (N, 3) where the circle is encoded as [X, Y, R]!");

	for(size_t i = 0; i < circlesXYR.shape[0]; i++)
	{
		auto circle = circlesXYR[i];
		drawCircle(tgt, {circle[0].item<float>(), circle[1].item<float>()}, circle[2].item<float>(), thickness, color);
	}
}

inline double Gauss(double x, double y, double mu, double sigma)
{
	const double xMinusMu = x-mu;
	const double yMinusMu = y-mu;
	const double sigmaSqr = sigma*sigma;

	return (1.0/(2.0*sigmaSqr*M_PI)) * std::exp(-(xMinusMu*xMinusMu + yMinusMu*yMinusMu)/(2.0*sigmaSqr));
}

inline Tensor MakeGaussKernel(DTYPE dtype, unsigned int kernelSize, double sigma, double mu)
{
	Tensor tgt = Tensor::empty({kernelSize, kernelSize}, dtype);
	for(uint y = 0; y < kernelSize; y++)
		for(uint x = 0; x < kernelSize; x++)
		{
			const int localX = x - kernelSize/2;
			const int localY = y - kernelSize/2;
			tgt[{y, x}] = Gauss(localX, localY, mu, sigma);
		}

	return tgt;
}

Tensor image::gaussianBlur(Tensor input, unsigned int kernelSize, double sigma)
{
	Tensor output = Tensor::empty_like(input);
	gaussianBlur(input, output, kernelSize, sigma);
	return output;
}

Tensor& image::gaussianBlur(Tensor input, Tensor& tgt, unsigned int kernelSize, double sigma)
{
	auto* backend = input.getBackendAs<cv::CVBackendTensor>();
	AssertExcept(backend, "A conv2d capable backend is required!");

	const auto kernel = MakeGaussKernel(tgt.dtype, kernelSize, sigma, 0).to(tgt.device);

	backend->conv2d(input, tgt, kernel, {1, 1}, {kernelSize/2, kernelSize/2}, {1, 1});

	return tgt;
}

Tensor image::sobelDerivative(Tensor input, bool horizontal)
{
	Tensor output = Tensor::empty_like(input);
	sobelDerivative(input, output, horizontal);
	return output;
}

Tensor& image::sobelDerivative(Tensor input, Tensor& tgt, bool horizontal)
{
	// Kernels
	static const Tensor s_horKernel({3, 3}, {
		1.0f, 0.0f, -1.0f,
		2.0f, 0.0f, -2.0f,
		1.0f, 0.0f, -1.0f
	});

	static const Tensor s_vertKernel({3, 3}, {
		1.0f, 2.0f, 1.0f,
		0.0f, 0.0f, 0.0f,
		-1.0f, -2.0f, -1.0f
	});

	auto* backend = input.getBackendAs<cv::CVBackendTensor>();
	AssertExcept(backend, "A conv2d capable backend is required!");

	Tensor kernel = (horizontal ? 
			s_horKernel.astype(input.dtype).to(input.device)
			: s_vertKernel.astype(input.dtype).to(input.device));

	backend->conv2d(input, tgt, kernel, {1, 1}, {1, 1}, {1, 1});
	return tgt;
}
