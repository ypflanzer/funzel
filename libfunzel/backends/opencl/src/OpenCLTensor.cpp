#include <funzel/Tensor.hpp>
#include "OpenCLTensor.hpp"
#include <clblast.h>
#include <cassert>

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
			const auto maxWorkgroup = tensor->m_device.device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
			
			::cl::NDRange realLocalSz = globalSz;
			for(size_t i = 0; i < globalSz.dimensions(); i++)
				realLocalSz.get()[i] = maxWorkgroup;
			
			tensor->m_cmdQueue.enqueueNDRangeKernel(kernels[dtype], dims, globalSz, realLocalSz, nullptr, &event);
		}
		else
			tensor->m_cmdQueue.enqueueNDRangeKernel(kernels[dtype], dims, globalSz, localSz, nullptr, &event);

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

	m_cmdQueue = backend.getCommandQueue(idx);
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
	m_cmdQueue.enqueueReadBuffer(m_buffer, CL_TRUE, 0, sz, dest.get());
	return dest;
}

#include <iostream>
void OpenCLTensor::empty(const void* buffer, size_t sz, const Shape& shape, DTYPE dtype)
{
	this->dtype = dtype;
	this->size = sz;

	sz *= dtypeSizeof(dtype);
	m_buffer = ::cl::Buffer(m_device.context, CL_MEM_READ_WRITE, sz);

	if(buffer)
	{
		// TODO: Can we do without the sync here? It seems to introduce a race condition
		m_cmdQueue.enqueueWriteBuffer(m_buffer, CL_FALSE, 0, sz, buffer, nullptr, &m_currentEvent);
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
	t->m_cmdQueue.enqueueCopyBuffer(m_buffer, t->m_buffer, 0, 0, sz, nullptr, &t->m_currentEvent);	
	return std::shared_ptr<BackendTensor>(t);
}

static const std::string s_fillKernel = R"krnl(
__kernel void Kernel(__global T* dest, ulong count, T value)
{
	ulong idx = get_global_id(0);
	if(idx < count)
	{
		dest[idx] = value;
	}
}
)krnl";

void OpenCLTensor::fill(const Tensor& self, double scalar)
{
	wait();
	switch(dtype)
	{
		case FLOAT32:
			m_cmdQueue.enqueueFillBuffer<float>(m_buffer, scalar, self.offset/sizeof(float), self.size()*sizeof(float), nullptr, &m_currentEvent);
		break;
		
		case FLOAT64:
			m_cmdQueue.enqueueFillBuffer<double>(m_buffer, scalar, self.offset/sizeof(double), self.size()*sizeof(double), nullptr, &m_currentEvent);
		break;
		default: ThrowError("Unsupported dtype!");
	}

	#if 0
	// Build kernels!
	static CLTemplateKernel kernel{s_fillKernel};
	wait();

	switch(dtype)
	{
		case FLOAT32: m_currentEvent = kernel.call(m_cmdQueue, {0}, {size}, {32}, dtype, m_buffer, size, static_cast<float>(scalar)); break;
		case FLOAT64: m_currentEvent = kernel.call(m_cmdQueue, {0}, {size}, {32}, dtype, m_buffer, size, static_cast<double>(scalar)); break;
		default: ThrowError("Unsupported dtype!");
	}
	#endif
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

	if(self.shape.size() > 1 && self.shape[1] > 1)
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
		case FLOAT32: {
			err = clblast::Axpy<float>(self.size(), float(alpha), abuf(), self.offset/sizeof(float), stride/sizeof(float),
										bbuf(), tgt.offset/sizeof(float), stride/sizeof(float),
										&tgtBackend->m_cmdQueue(), &tgtBackend->m_currentEvent());
		} break;

		case FLOAT64: {
			err = clblast::Axpy<double>(self.size(), double(alpha), abuf(), self.offset/sizeof(double), stride/sizeof(double),
										bbuf(), tgt.offset/sizeof(double), stride/sizeof(double),
										&tgtBackend->m_cmdQueue(), &tgtBackend->m_currentEvent());
		} break;
		default: ThrowError("Unsupported dtype!");
	}

	AssertExcept(err == clblast::StatusCode::kSuccess, "CLBlast error: " + std::to_string((int) err));
}

void OpenCLTensor::sub(const Tensor& self, const Tensor& b, double alpha)
{

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
	AssertExcept(self.shape[1] == b.shape[0], "A");
	//AssertExcept(self.shape[0] == tgt.shape[0] && tgt.shape[1] == b.shape[1], "B");

	auto m = tgt.shape[0];
	auto k = self.shape[1];
	auto n = b.shape[1];

	switch(dtype)
	{
		case FLOAT32: {
			err = clblast::Gemm<float>(
				clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,
				m, n, k, 1.0f,
				abuf(), self.offset/sizeof(float), k,
				bbuf(), b.offset/sizeof(float), n,
				1.0f,
				cbuf(), tgt.offset/sizeof(float), n,
				&tgtBackend->m_cmdQueue(), &tgtBackend->m_currentEvent());
		} break;
		case FLOAT64: {
			err = clblast::Gemm<double>(
				clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,
				m, n, k, 1.0f,
				abuf(), self.offset/sizeof(double), k,
				bbuf(), b.offset/sizeof(double), n,
				1.0f,
				cbuf(), tgt.offset/sizeof(double), n,
				&tgtBackend->m_cmdQueue(), &tgtBackend->m_currentEvent());
		} break;
		default: ThrowError("Unsupported dtype!");
	}

	AssertExcept(err == clblast::StatusCode::kSuccess, "CLBlast error: " + std::to_string((int) err));
}

