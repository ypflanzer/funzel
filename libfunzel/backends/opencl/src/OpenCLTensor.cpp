#include <funzel/Tensor.hpp>
#include "OpenCLTensor.hpp"
#include <clblast.h>
#include <cassert>

#include <spdlog/spdlog.h>
#include <iostream>

#include <funzel/cv/Image.hpp>

using namespace funzel;
using namespace funzel::cl;

struct funzel::cl::CLTemplateKernel
{
	// Build kernels!
	CLTemplateKernel(const std::string& src):
		src(src) {}

	CLTemplateKernel(std::string&& src):
		src(std::move(src)) {}

	template<int i, typename... Args, typename Arg>
	void setArgs(DTYPE dtype, Arg&& arg, Args&&... args)
	{
		kernels[dtype].setArg(i, arg);
		setArgs<i+1>(dtype, args...);
	}

	template<int i>
	void setArgs(DTYPE dtype)
	{}

	static inline size_t padGlobalSize(size_t global, size_t local)
	{
		return ((global + local - 1) / local) * local;
	}

	template<typename... Args>
	::cl::Event call(
		OpenCLTensor* tensor,
		const ::cl::NDRange& dims,
		const ::cl::NDRange& globalSz,
		const ::cl::NDRange& localSz,
		DTYPE dtype, Args&&... args)
	{
		auto& backend = OpenCLBackend::the();

		if(kernels[dtype].get() == nullptr)
		{
			kernels[dtype] = backend.buildKernel(tensor->m_device, src, dtype);
		}

		setArgs<0>(dtype, args...);

		::cl::Event event;

		if(localSz[0] == 0)
		{
			const unsigned int maxWorkgroup = std::pow(tensor->m_device.device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), 1.0/globalSz.dimensions());
			
			::cl::NDRange realLocalSz = globalSz;
			::cl::NDRange realGlobalSz = globalSz;
			for(size_t i = 0; i < globalSz.dimensions(); i++)
			{
				realLocalSz.get()[i] = maxWorkgroup;
				realGlobalSz.get()[i] = padGlobalSize(globalSz[i], maxWorkgroup);
			}

			tensor->m_device.queue.enqueueNDRangeKernel(kernels[dtype], dims, realGlobalSz, realLocalSz, nullptr, &event);
		}
		else
			tensor->m_device.queue.enqueueNDRangeKernel(kernels[dtype], dims, globalSz, localSz, nullptr, &event);

		return event;
	}

	std::string src;
	::cl::Kernel kernels[DTYPE_MAX];
};

OpenCLTensor::OpenCLTensor():
	OpenCLTensor("0") {}

OpenCLTensor::OpenCLTensor(const std::string& args):
	m_clArgs(args)
{
	auto& backend = OpenCLBackend::the();
	size_t idx = 0;

	if(!args.empty())
		idx = std::stol(args);

	m_device = backend.getDevice(idx);
}

void OpenCLTensor::wait()
{
	if(m_currentEvent.get())
	{
		auto err = m_currentEvent.wait();
		m_currentEvent = nullptr;

		AssertExcept(err == CL_SUCCESS, "Could not wait for OpenCL events: " + std::to_string(err));
	}
}

void* OpenCLTensor::data(size_t offset)
{
	ThrowError("Cannot directly access GPU memory of OpenCL device!");
}

std::shared_ptr<char> OpenCLTensor::buffer()
{
	size_t sz = dtypeSizeof(dtype)*size;
	auto dest = std::shared_ptr<char>((char*) std::malloc(sz));

	wait();
	m_device.queue.enqueueReadBuffer(m_buffer, CL_TRUE, 0, sz, dest.get());
	return dest;
}

void OpenCLTensor::empty(const void* buffer, size_t sz, const Shape& shape, DTYPE dtype)
{
	this->dtype = dtype;
	this->size = sz;

	sz *= dtypeSizeof(dtype);
	m_buffer = ::cl::Buffer(m_device.context, CL_MEM_READ_WRITE, sz);
	AssertExcept(m_buffer.get(), "Could not allocated GPU buffer!");

	if(buffer)
	{
		// TODO: Can we do without the sync here? It seems to introduce a race condition
		m_device.queue.enqueueWriteBuffer(m_buffer, CL_FALSE, 0, sz, buffer, nullptr, &m_currentEvent);
		wait();
	}
}

void OpenCLTensor::empty(std::shared_ptr<char> buffer, size_t sz, const Shape& shape, DTYPE dtype)
{
	empty((const void*) buffer.get(), sz, shape, dtype);
}

std::shared_ptr<BackendTensor> OpenCLTensor::clone() const
{
	OpenCLTensor* t = new OpenCLTensor(m_clArgs);
	size_t sz = size*dtypeSizeof(dtype);

	t->empty(nullptr, sz, {size}, dtype);

	t->wait();
	t->m_device.queue.enqueueCopyBuffer(m_buffer, t->m_buffer, 0, 0, sz, nullptr, &t->m_currentEvent);	
	return std::shared_ptr<BackendTensor>(t);
}

void OpenCLTensor::fill(const Tensor& self, double scalar)
{
	wait();
	switch(dtype)
	{
		case DFLOAT32:
			m_device.queue.enqueueFillBuffer<float>(m_buffer, scalar, self.offset/sizeof(float), self.size()*sizeof(float), nullptr, &m_currentEvent);
		break;
		
		case DFLOAT64:
			m_device.queue.enqueueFillBuffer<double>(m_buffer, scalar, self.offset/sizeof(double), self.size()*sizeof(double), nullptr, &m_currentEvent);
		break;
		default: ThrowError("Unsupported dtype!");
	}
}

double OpenCLTensor::sum(const Tensor& self)
{
	return 0;
}

template<typename Fn>
::cl::Event OpenCLTensor::DoStrided(const Tensor& self, CLTemplateKernel& kernel, Fn&& fn)
{
	if(self.shape.empty())
		return ::cl::Event();
	
	if(!(self.flags & C_CONTIGUOUS) && self.shape.size() > 1)
	{
		::cl::Event event;
		for(int i = 0; i < self.shape[0]; i++)
		{
			event = DoStrided(self[i], kernel, fn);
		}

		return event;
	}

	const size_t tsize = dtypeSizeof(dtype);
	size_t stride = self.strides.back()/tsize;
	size_t offset = self.offset/tsize;
	size_t size = self.size();
	return fn(this, self, stride, offset, size);
}

inline OpenCLTensor* getCLTensor(Tensor& t)
{
	auto* tgtBackend = t.getBackendAs<OpenCLTensor>();
	AssertExcept(tgtBackend, "Invalid backend for operation!");
	return tgtBackend;
}

inline OpenCLTensor* getCLTensor(const Tensor& t)
{
	auto* tgtBackend = t.getBackendAs<OpenCLTensor>();
	AssertExcept(tgtBackend, "Invalid backend for operation!");
	return tgtBackend;
}

size_t OpenCLTensor::workgroupSize() const
{
	return m_device.device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
}

static const std::string s_absKernel = R"krnl(
__kernel void Kernel(__global const T* src, __global T* dest, ulong offset, ulong stride, ulong count)
{
	const ulong idx = get_global_id(0);
	if(idx < count)
	{
		const ulong gid = offset + idx*stride;
		const T elem = src[gid];
		dest[gid] = (elem < 0 ? -elem : elem); // Builtin abs is only defined for integer types
	}
}
)krnl";

void OpenCLTensor::abs(const Tensor& self, Tensor tgt)
{
	// Build kernels!
	static CLTemplateKernel kernel{s_absKernel};
	auto* tgtBackend = getCLTensor(tgt);
	
	tgtBackend->wait();
	wait();

	const auto Abs = [tgtBackend](OpenCLTensor* clt, const Tensor& self, size_t stride, size_t offset, size_t count) {
		return kernel.call(tgtBackend, {0}, {count}, {0}, self.dtype, tgtBackend->m_buffer, clt->m_buffer, offset, stride, count);
	};

	m_currentEvent = DoStrided(self, kernel, Abs);
}

static const std::string s_unaryKernel = R"krnl(
__kernel void Kernel(__global const T* src, __global T* dest, ulong offset, ulong stride, ulong count)
{
	const ulong idx = get_global_id(0);
	if(idx < count)
	{
		const ulong gid = offset + idx*stride;
		const T elem = src[gid];
		dest[gid] = OP(elem);
	}
}
)krnl";

#define DEFINE_UNARY_OP(op) \
void OpenCLTensor::op(const Tensor& self, Tensor tgt) \
{ \
	static CLTemplateKernel kernel{"#define OP " #op + s_unaryKernel}; \
	auto* tgtBackend = getCLTensor(tgt); \
	tgtBackend->wait(); \
	wait(); \
	const auto fn = [tgtBackend](OpenCLTensor* clt, const Tensor& self, size_t stride, size_t offset, size_t count) { \
		return kernel.call(tgtBackend, {0}, {count}, {0}, self.dtype, tgtBackend->m_buffer, clt->m_buffer, offset, stride, count); \
	}; \
	m_currentEvent = DoStrided(self, kernel, fn); \
}

DEFINE_UNARY_OP(exp)
DEFINE_UNARY_OP(sqrt)
DEFINE_UNARY_OP(sin)
DEFINE_UNARY_OP(cos)
DEFINE_UNARY_OP(tan)
DEFINE_UNARY_OP(tanh)

void OpenCLTensor::mulAdd(const Tensor& self, Tensor tgt, double alpha)
{
	if(self.shape.empty())
		return;

	auto* tgtBackend = tgt.getBackendAs<OpenCLTensor>();
	AssertExcept(tgtBackend, "Invalid backend for operation!");

	assert(self.getBackendAs<OpenCLTensor>() == this);

	// Wait for both tensors to be available
	wait();
	tgtBackend->wait();

	// We can handle contiguous memory at once, without subdividing further.
	//if(self.shape.size() > 1 && self.shape[1] > 1)
	if(!(self.flags & C_CONTIGUOUS) && self.shape.size() > 1)
	{
		for(int i = 0; i < self.shape[0]; i++)
		{
			mulAdd(self[i], tgt[i], alpha);
		}

		return;
	}

	::cl::Buffer abuf = m_buffer;
	::cl::Buffer bbuf = tgtBackend->m_buffer;
	size_t stride = self.strides.back();

	clblast::StatusCode err;
	switch(dtype)
	{
		case DFLOAT32: {
			err = clblast::Axpy<float>(self.size(), float(alpha), abuf(), self.offset/sizeof(float), stride/sizeof(float),
										bbuf(), tgt.offset/sizeof(float), stride/sizeof(float),
										&tgtBackend->m_device.queue(), &tgtBackend->m_currentEvent());
		} break;

		case DFLOAT64: {
			err = clblast::Axpy<double>(self.size(), double(alpha), abuf(), self.offset/sizeof(double), stride/sizeof(double),
										bbuf(), tgt.offset/sizeof(double), stride/sizeof(double),
										&tgtBackend->m_device.queue(), &tgtBackend->m_currentEvent());
		} break;
		default: ThrowError("Unsupported dtype!");
	}

	AssertExcept(err == clblast::StatusCode::kSuccess, "CLBlast error: " + std::to_string((int) err));
}

void OpenCLTensor::sub(const Tensor& self, const Tensor& b, double alpha)
{

}

static const std::string s_divKernel = R"krnl(
__kernel void Kernel(
	ulong count,
	__global const T* a, ulong aOffset, ulong aStride,
	__global const T* b, ulong bOffset, ulong bStride,
	__global T* c, ulong cOffset, ulong cStride)
{
	const ulong idx = get_global_id(0);
	if(idx < count)
	{
		const ulong aidx = aOffset + idx*aStride;
		const ulong bidx = bOffset + idx*bStride;
		const ulong cidx = cOffset + idx*cStride;

		c[cidx] = a[aidx] / b[bidx];
	}
}
)krnl";

void OpenCLTensor::div(const Tensor& self, const Tensor& b, Tensor tgt)
{
	// Build kernels!
	static CLTemplateKernel kernel{s_divKernel};
	auto* bBackend = getCLTensor(b);
	auto* cBackend = getCLTensor(tgt);

	const size_t cOffset = tgt.offset/dtypeSizeof(dtype);
	const size_t bOffset = b.offset/dtypeSizeof(dtype);

	const size_t cStride = tgt.strides.back()/dtypeSizeof(dtype);
	const size_t bStride = b.strides.back()/dtypeSizeof(dtype);

	bBackend->wait();
	cBackend->wait();
	wait();

	const auto Div = [bBackend, cBackend, cOffset, bOffset, cStride, bStride]
		(OpenCLTensor* aBackend, const Tensor& self, size_t stride, size_t offset, size_t count) {
			return kernel.call(cBackend, {0}, {count}, {0}, self.dtype,
					count,
					aBackend->m_buffer, offset, stride,
					bBackend->m_buffer, bOffset, bStride,
					cBackend->m_buffer, cOffset, cStride);
	};

	m_currentEvent = DoStrided(self, kernel, Div);
}

void OpenCLTensor::matmul(const Tensor& self, Tensor b, Tensor tgt)
{
	if(self.shape.empty())
		return;

	auto* bBackend = b.getBackendAs<OpenCLTensor>();
	auto* tgtBackend = tgt.getBackendAs<OpenCLTensor>();

	AssertExcept(tgtBackend && bBackend, "Invalid backend for operation!");
	AssertExcept(self.getBackend() != tgt.getBackend(), "Cannot multiply matrices in-place!");
	assert(self.getBackend() == this && "Wrong tensor given first!");

	// Wait for both tensors to be available
	wait();
	tgtBackend->wait();

	if(self.shape.size() > 2)
	{
		for(int i = 0; i < self.shape[0]; i++)
		{
			matmul(self[i], b[i], tgt[i]);
		}

		return;
	}

	::cl::Buffer abuf = m_buffer;
	::cl::Buffer bbuf = bBackend->m_buffer;
	::cl::Buffer cbuf = tgtBackend->m_buffer;

	clblast::StatusCode err;
	AssertExcept(self.shape[1] == b.shape[0], "Invalid shapes for matrix multiplication!");

	auto m = tgt.shape[0];
	auto k = self.shape[1];
	auto n = b.shape[1];

	switch(dtype)
	{
		case DFLOAT32: {
			err = clblast::Gemm<float>(
				clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,
				m, n, k, 1.0f,
				abuf(), self.offset/sizeof(float), k,
				bbuf(), b.offset/sizeof(float), n,
				1.0f,
				cbuf(), tgt.offset/sizeof(float), n,
				&tgtBackend->m_device.queue(), &tgtBackend->m_currentEvent());
		} break;
		case DFLOAT64: {
			err = clblast::Gemm<double>(
				clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,
				m, n, k, 1.0f,
				abuf(), self.offset/sizeof(double), k,
				bbuf(), b.offset/sizeof(double), n,
				1.0f,
				cbuf(), tgt.offset/sizeof(double), n,
				&tgtBackend->m_device.queue(), &tgtBackend->m_currentEvent());
		} break;
		default: ThrowError("Unsupported dtype!");
	}

	AssertExcept(err == clblast::StatusCode::kSuccess, "CLBlast error: " + std::to_string((int) err));
}

static const std::string s_poolingKernel = R"krnl(
__kernel void Kernel(
	ulong inputCount,
	__global const T* input, ulong aOffset, ulong aStrideX, ulong aStrideY,
	ulong tgtWidth, ulong tgtHeight,
	__global T* output, ulong tgtOffset, ulong tgtStrideX, ulong tgtStrideY,
	int kernelW, int kernelH,
	int dilationX, int dilationY,
	int strideX, int strideY)
{
	const ulong yidx = get_global_id(0);
	const ulong xidx = get_global_id(1);

	if(xidx < tgtWidth && yidx < tgtHeight)
	{
		T accum = 0;

		const ulong inX = xidx*strideX;
		const ulong inY = yidx*strideY;

		const ulong yinOff = inY*aStrideY;
		const ulong xinOff = inX*aStrideX;

		const ulong xoutOff = xidx*tgtStrideX;
		const ulong youtOff = yidx*tgtStrideY;

		const int halfKw = (int)(kernelW/2);
		const int halfKh = (int)(kernelH/2);
		const int ksize = kernelW*kernelH;

		// TODO Optimize!
		for(int ky = -halfKh; ky <= halfKh; ky++)
		{
			for(int kx = -halfKw; kx <= halfKw; kx++)
			{
				const int dkx = dilationX*kx;
				const int dky = dilationY*ky;

				const long inputOffset = yinOff + xinOff + dkx*aStrideX + dky*aStrideY;
				if(inputOffset >= 0 && inputOffset < inputCount)
				{
					accum = ACCUM(accum, input[inputOffset], ksize);
				}
			}
		}

		output[youtOff + xoutOff] = accum;
	}
}
)krnl";

void OpenCLTensor::pool2d(
			const Tensor& self, Tensor tgt,
			POOLING_MODE mode,
			const UVec2& kernelSize,
			const UVec2& stride,
			const UVec2& padding,
			const UVec2& dilation)
{
	if(self.shape.empty())
		return;

	static CLTemplateKernel maxkernel{"#define ACCUM(accum, b, count) max(accum, b)\n" + s_poolingKernel};
	static CLTemplateKernel meankernel{"#define ACCUM(accum, b, count) ((accum) + ((b)/(count)))\n" + s_poolingKernel};

	auto* tgtBackend = tgt.getBackendAs<OpenCLTensor>();

	AssertExcept(self.getBackend() != tgt.getBackend(), "Cannot apply pooling in-place!");
	assert(self.getBackend() == this && "Wrong tensor given first!");

	// Wait for both tensors to be available
	wait();
	tgtBackend->wait();

	if(self.shape.size() > 2)
	{
		for(int i = 0; i < self.shape[0]; i++)
		{
			pool2d(self[i], tgt[i], mode, kernelSize, stride, padding, dilation);
		}

		return;
	}

	::cl::Buffer abuf = m_buffer;
	::cl::Buffer cbuf = tgtBackend->m_buffer;

	const int64_t outstrideY = tgt.strides[0]/dtypeSizeof(self.dtype);
	const int64_t outstrideX = tgt.strides[1]/dtypeSizeof(self.dtype);
	const int64_t instrideY = self.strides[0]/dtypeSizeof(self.dtype);
	const int64_t instrideX = self.strides[1]/dtypeSizeof(self.dtype);

	CLTemplateKernel& kernel = (mode == MEAN_POOLING ? meankernel : maxkernel);
	kernel.call(tgtBackend, ::cl::NullRange, {tgt.shape[0], tgt.shape[1]}, ::cl::NullRange, self.dtype,
				self.shape[0]*self.shape[1],
				abuf, self.offset/dtypeSizeof(self.dtype), instrideX, instrideY,
				tgt.shape[1], tgt.shape[0],
				cbuf, tgt.offset/dtypeSizeof(tgt.dtype), outstrideX, outstrideY,
				kernelSize[0], kernelSize[1],
				dilation[0], dilation[1],
				stride[0], stride[1]);
}

template<typename T>
static inline void Conv2DInner(
	const Tensor& self, Tensor tgt,
	const Tensor& kernel,
	const UVec2& stride,
	const UVec2& padding,
	const UVec2& dilation,
	size_t finalDimensions,
	size_t width, size_t height, size_t batch, size_t channels)
{
	
	if (self.shape.size() > finalDimensions)
	{
		for (int i = 0; i < self.shape[0]; i++)
		{
			Conv2DInner<T>(self[i], tgt[i], kernel, stride, padding, dilation, finalDimensions, width, height, batch, channels);
		}
		return;
	}

	auto* tgtBackend = tgt.getBackendAs<OpenCLTensor>();
	auto* krnlBackend = kernel.getBackendAs<OpenCLTensor>();
	auto* selfBackend = self.getBackendAs<OpenCLTensor>();

	auto& tgtBuffer = tgtBackend->clbuffer();
	auto& selfBuffer = selfBackend->clbuffer();
	auto& krnlBuffer = krnlBackend->clbuffer();

	auto status = clblast::Convgemm<T>(clblast::KernelMode::kCrossCorrelation,
			channels,
			height, width, // width, height
			kernel.shape[0], kernel.shape[1], // kernelWidth, kernelHeight,
			padding[0], padding[1],
			stride[0], stride[1],
			dilation[0], dilation[1],
			1, // 1 kernel only. Maybe that can be more?
			
			batch, // Batch size
			selfBuffer(), self.offset / sizeof(T),
			krnlBuffer(), kernel.offset / sizeof(T),
			tgtBuffer(), tgt.offset / sizeof(T),

			&tgtBackend->clcmdQueue()(), &tgtBackend->clcurrentEvent()());


	AssertExcept(status == clblast::StatusCode::kSuccess, "CLBlast Error: " + std::to_string((int) status));
}

#if 1
void OpenCLTensor::conv2d(
	const Tensor& self, Tensor tgt,
	const Tensor& kernel,
	const UVec2& stride,
	const UVec2& padding,
	const UVec2& dilation)
{
	if (self.shape.empty())
		return;

	AssertExcept(self.shape.size() >= 2,
		"A 2D convolutions requires a tensor of type BCHW, CHW or HW, the given tensor has only one dimension.");

	size_t finalDimensions = 2;
	size_t width = 0, height = 0, outWidth = 0, outHeight = 0;
	size_t batch = 1, channels = 1;

	const auto rshape = self.shape.rbegin();
	const auto outRshape = tgt.shape.rbegin();

	// Check if input can be processed as one batch
	if(self.shape.size() >= 4)
	{
		finalDimensions = 4;
		batch = rshape[3];

		channels = rshape[2];
		height = rshape[1];
		width = rshape[0];

		outHeight = outRshape[1];
		outWidth = outRshape[0];
	}
	else if(self.shape.size() == 3)
	{
		finalDimensions = 3;
		batch = 1;

		channels = rshape[2];
		height = rshape[1];
		width = rshape[0];

		outHeight = outRshape[1];
		outWidth = outRshape[0];
	}
	else if(self.shape.size() == 2)
	{
		finalDimensions = 2;
		batch = 1;
		channels = 1;
		height = rshape[1];
		width = rshape[0];

		outHeight = outRshape[1];
		outWidth = outRshape[0];
	}

	auto* tgtBackend = tgt.getBackendAs<OpenCLTensor>();
	auto* kernelBackend = kernel.getBackendAs<OpenCLTensor>();

	AssertExcept(self.getBackend() != tgt.getBackend(), "Cannot apply convolution in-place!");
	assert(self.getBackend() == this && "Wrong tensor given first!");

	const size_t oh = ((height + size_t(2) * padding[0] - dilation[0] * (kernel.shape[0] - 1) - 1) / stride[0]) + 1;
	const size_t ow = ((width + size_t(2) * padding[1] - dilation[1] * (kernel.shape[1] - 1) - 1) / stride[1]) + 1;
	AssertExcept(outHeight == oh && outWidth == ow, "Invalid output size: " << outHeight << " != " << oh << " or " << outWidth << " != " << ow);

	AssertExcept((channels == 1 && kernel.shape.size() == 2) || kernel.shape[2] == channels,
					"An image with CHW needs a kernel with HWC, the number of channels must match!");

	// Wait for both tensors to be available
	wait();
	tgtBackend->wait();
	kernelBackend->wait();

	switch (dtype)
	{
	case DFLOAT32: {
		Conv2DInner<float>(
			self, tgt,
			kernel,
			stride,
			padding,
			dilation,
			finalDimensions,
			width, height, batch, channels);
	} break;

	case DFLOAT64: {
		Conv2DInner<double>(
			self, tgt,
			kernel,
			stride,
			padding,
			dilation,
			finalDimensions,
			width, height, batch, channels);
	} break;
	default: ThrowError("Unsupported dtype!");
	}
}
#endif
#if 0
void OpenCLTensor::conv2d(
	const Tensor& self, Tensor tgt,
	const Tensor& kernel,
	const UVec2& stride,
	const UVec2& padding,
	const UVec2& dilation)
{
	if (self.shape.empty())
		return;

	auto* tgtBackend = tgt.getBackendAs<OpenCLTensor>();
	auto* krnlBackend = kernel.getBackendAs<OpenCLTensor>();
	auto* selfBackend = self.getBackendAs<OpenCLTensor>();

	AssertExcept(tgtBackend && krnlBackend && selfBackend, "At least one of the input tensors is not an OpenCL tensor!");

	AssertExcept(self.getBackend() != tgt.getBackend(), "Cannot apply convolution in-place!");
	assert(self.getBackend() == this && "Wrong tensor given first!");

	// Wait for both tensors to be available
	wait();
	tgtBackend->wait();
	krnlBackend->wait();

	if (self.shape.size() > 2)
	{
		for (int i = 0; i < self.shape[0]; i++)
		{
			conv2d(self[i], tgt[i], kernel, stride, padding, dilation);
		}

		return;
	}

	const size_t oh = ((self.shape[0] + size_t(2) * padding[0] - dilation[0] * (kernel.shape[0] - 1) - 1) / stride[0]) + 1;
	const size_t ow = ((self.shape[1] + size_t(2) * padding[1] - dilation[1] * (kernel.shape[1] - 1) - 1) / stride[1]) + 1;
	AssertExcept(tgt.shape[0] == oh && tgt.shape[1] == ow, "Invalid output size: " << tgt.shape[0] << " != " << oh << " or " << tgt.shape[1] << " != " << ow);

	const size_t channels = 1;
	const size_t kernels = 1; // (kernel.shape.size() > 2 ? kernel.shape[2] : 1);

	switch (dtype)
	{
	case DFLOAT32: {
		clblast::Convgemm<float>(clblast::KernelMode::kCrossCorrelation, channels,
			self.shape[0], self.shape[1], // width, height
			kernel.shape[0], kernel.shape[1], // kernelWidth, kernelHeight,
			padding[0], padding[1],
			stride[0], stride[1],
			dilation[0], dilation[1],
			kernels,
			
			1, // Batch size
			selfBackend->m_buffer(), self.offset / sizeof(float),
			krnlBackend->m_buffer(), kernel.offset / sizeof(float),
			tgtBackend->m_buffer(), tgt.offset / sizeof(float),

			&tgtBackend->m_device.queue(), &tgtBackend->m_currentEvent());
	} break;

	case DFLOAT64: {
		clblast::Convgemm<double>(clblast::KernelMode::kCrossCorrelation, 1,
			self.shape[0], self.shape[1], // width, height
			kernel.shape[0], kernel.shape[1], // kernelWidth, kernelHeight,
			padding[0], padding[1],
			stride[0], stride[1],
			dilation[0], dilation[1],
			(kernel.shape.size() > 2 ? kernel.shape[2] : 1),

			1, // Batch size
			self.getBackendAs<OpenCLTensor>()->m_buffer(), self.offset / sizeof(double),
			krnlBackend->m_buffer(), kernel.offset / sizeof(double),
			tgtBackend->m_buffer(), tgt.offset / sizeof(double),

			&tgtBackend->m_device.queue(), &tgtBackend->m_currentEvent());
	} break;
	default: ThrowError("Unsupported dtype!");
	}
}
#endif
static const std::string s_reluKernel = R"krnl(
__kernel void Kernel(__global const T* src, __global T* dest, ulong offset, ulong stride, ulong count, double slope)
{
	const ulong idx = get_global_id(0);
	if(idx < count)
	{
		const ulong gid = offset + idx*stride;
		const T elem = src[gid];
		dest[gid] = (elem < 0 ? elem*slope : elem);
	}
}
)krnl";

void OpenCLTensor::relu(const Tensor& self, Tensor& tgt, double negativeSlope)
{
	// Build kernels!
	static CLTemplateKernel kernel{s_reluKernel};
	auto* tgtBackend = getCLTensor(tgt);
	
	tgtBackend->wait();
	wait();

	const auto Relu = [tgtBackend, negativeSlope](OpenCLTensor* clt, const Tensor& self, size_t stride, size_t offset, size_t count) {
		return kernel.call(tgtBackend, {0}, {count}, {0}, self.dtype, tgtBackend->m_buffer, clt->m_buffer, offset, stride, count, negativeSlope);
	};

	m_currentEvent = DoStrided(self, kernel, Relu);
}
