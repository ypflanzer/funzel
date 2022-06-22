#include <funzel/Tensor.hpp>
#include "BlasTensor.hpp"

#include <iostream>
#include <cblas.h>
#include <cstring>
#include <cmath>

#ifdef WIN32
#include <windows.h>
static size_t systemMemory()
{
	MEMORYSTATUSEX status;
	status.dwLength = sizeof(status);
	
	GlobalMemoryStatusEx(&status);
	return status.ullTotalPhys;
}
#else
#include <unistd.h>
static size_t systemMemory()
{
	size_t pageCount = sysconf(_SC_PHYS_PAGES);
	size_t pageSize = sysconf(_SC_PAGE_SIZE);
	return pageCount * pageSize;
}
#endif

using namespace funzel;
using namespace blas;

void BlasTensor::initializeBackend()
{
	DeviceProperties props;
	props.deviceID = "BLAS";
	props.deviceName = "Default CPU device";
	props.vendorName = "Funzel";
	props.memorySize = systemMemory();
	props.isGPU = false;

	backend::RegisterDevice(props);
}

void* BlasTensor::data(size_t offset)
{
	// TODO: Bounds check!
	const auto sz = size*dtypeSizeof(dtype);
	AssertExcept(offset < sz, "Out of bounds access: " + std::to_string(offset) + " >= " + std::to_string(sz));
	return m_data.get() + offset;
}

void BlasTensor::empty(std::shared_ptr<char> buffer, size_t sz, const Shape& shape, DTYPE dtype)
{
	this->size = sz;
	this->dtype = dtype;

	if(!buffer)
	{
		// m_data = std::shared_ptr<char>((char*) std::aligned_alloc(8, size()*dtypeSizeof(dtype)));
		m_data = std::shared_ptr<char>((char*) std::malloc(size*dtypeSizeof(dtype)));
	}
	else
	{
		m_data = buffer;
	}
}

void BlasTensor::empty(const void* buffer, size_t sz, const Shape& shape, DTYPE dtype)
{
	this->size = sz;
	this->dtype = dtype;

	m_data = std::shared_ptr<char>((char*) std::malloc(size*dtypeSizeof(dtype)));

	if(buffer)
	{
		memcpy(m_data.get(), buffer, this->size*dtypeSizeof(dtype));
	}
}

std::shared_ptr<BackendTensor> BlasTensor::clone() const
{
	BlasTensor* t = new BlasTensor;
	size_t sz = size*dtypeSizeof(dtype);
	auto data = std::shared_ptr<char>((char*) std::malloc(sz));

	memcpy(data.get(), m_data.get(), sz);

	t->empty(data, sz, {size}, dtype);
	return std::shared_ptr<BackendTensor>(t);
}

static void CopyTensor(Tensor src, Tensor dest)
{
	if(src.shape.empty())
		return;

	if(src.shape.size() == 1)
	{
		#pragma omp parallel for
		for(int64_t i = 0; i < src.shape[0]; i++)
		{
			dest[i] = src[i].item<double>();
		}
		return;
	}

	for(size_t i = 0; i < src.shape[0]; i++)
	{
		CopyTensor(src[i], dest[i]);
	}
}

void BlasTensor::set(Tensor& self, const Tensor& src)
{
	CopyTensor(src, self);
}

template<typename T>
void Fill(void* data, T value, size_t count)
{
	#pragma omp parallel for
	for(int64_t i = 0; i < count; i++)
	{
		reinterpret_cast<T*>(data)[i] = value;
	}
}

void BlasTensor::fill(const Tensor& self, double scalar)
{
	const auto sz = self.size();
	switch(self.dtype)
	{
		case INT32: Fill<int32_t>(m_data.get() + self.offset, scalar, sz); break;
		case INT64: Fill<int64_t>(m_data.get() + self.offset, scalar, sz); break;
		case FLOAT32: Fill<float>(m_data.get() + self.offset, scalar, sz); break;
		case FLOAT64: Fill<double>(m_data.get() + self.offset, scalar, sz); break;
		case UINT32: Fill<uint32_t>(m_data.get() + self.offset, scalar, sz); break;
		case UINT64: Fill<uint64_t>(m_data.get() + self.offset, scalar, sz); break;
		case BYTE: Fill<char>(m_data.get() + self.offset, scalar, sz); break;
		case UBYTE: Fill<unsigned char>(m_data.get() + self.offset, scalar, sz); break;
		default: ThrowError("Uknown dtype!");
	}
}

#include <iostream>

double BlasTensor::sum(const Tensor& self)
{
	if(self.shape.empty())
		return 0;
	
	if(self.shape.size() > 1)
	{
		double s = 0;
		#pragma omp parallel for
		for(int i = 0; i < self.shape[0]; i++)
		{
			#pragma omp atomic
			s += sum(self[i]);
		}

		return s;
	}

	void* data = m_data.get() + self.offset;
	size_t stride = self.strides.back();

	switch(dtype)
	{
		case FLOAT32: {
			const float one = 1.0f;
			return cblas_sdot(self.size(), reinterpret_cast<float*>(data), stride/sizeof(float), &one, 0);
		}
		case FLOAT64: {
			const double one = 1.0;
			return cblas_ddot(self.size(), reinterpret_cast<double*>(data), stride/sizeof(double), &one, 0);
		}
		default: ThrowError("Unsupported dtype!");
	}

	return 0;
}

template<typename T, typename Fn>
inline void TensorOpInner(const Tensor& self, Tensor& tgt, Fn op)
{
	for(size_t x = 0; x < self.shape[0]; x++)
	{
		T& v = self[x].ritem<T>();
		tgt[x].ritem<T>() = op(v);
	}
}

template<typename T, typename Fn>
inline void TensorOpOuter(const Tensor& self, Tensor tgt, Fn op)
{
	if(self.shape.size() > 1)
	{
		#pragma omp parallel for
		for(int i = 0; i < self.shape[0]; i++)
			TensorOpOuter<T>(self[i], tgt[i], op);

		return;
	}

	TensorOpInner<T>(self, tgt, op);
}

template<typename Fn>
inline void TensorOp(const Tensor& self, Tensor& tgt, Fn op)
{
	switch(self.dtype)
	{
		case FLOAT32: {
			TensorOpOuter<float>(self, tgt, op);
		} break;
		case FLOAT64: {
			TensorOpOuter<double>(self, tgt, op);
		} break;
		default: ThrowError("Unsupported dtype!");
	}
}

template<typename T>
inline void TensorAbs(const Tensor& self, Tensor& tgt)
{
	for(size_t x = 0; x < self.shape[0]; x++)
	{
		T& v = self[x].ritem<T>();
		tgt[x].ritem<T>() = std::abs(v);
	}
}

void BlasTensor::abs(const Tensor& self, Tensor tgt)
{
	TensorOp(self, tgt, [](const auto& v) { return std::abs(v); });
	#if 0
	if(self.shape.size() > 1)
	{
		#pragma omp parallel for
		for(int i = 0; i < self.shape[0]; i++)
			abs(self[i], tgt[i]);

		return;
	}

	switch(dtype)
	{
		case FLOAT32: {
			TensorAbs<float>(self, tgt);
		} break;
		case FLOAT64: {
			TensorAbs<double>(self, tgt);
		} break;
		default: ThrowError("Unsupported dtype!");
	}
	#endif
}

void BlasTensor::exp(const Tensor& self, Tensor tgt)
{
	TensorOp(self, tgt, [](const auto& v) { return std::exp(v); });
}

void BlasTensor::sqrt(const Tensor& self, Tensor tgt)
{
	TensorOp(self, tgt, [](const auto& v) { return std::sqrt(v); });
}

void BlasTensor::sin(const Tensor& self, Tensor tgt)
{
	TensorOp(self, tgt, [](const auto& v) { return std::sin(v); });
}

void BlasTensor::cos(const Tensor& self, Tensor tgt)
{
	TensorOp(self, tgt, [](const auto& v) { return std::cos(v); });
}

void BlasTensor::tan(const Tensor& self, Tensor tgt)
{
	TensorOp(self, tgt, [](const auto& v) { return std::tan(v); });
}

void BlasTensor::tanh(const Tensor& self, Tensor tgt)
{
	TensorOp(self, tgt, [](const auto& v) { return std::tanh(v); });
}

void BlasTensor::mulAdd(const Tensor& self, Tensor tgt, double alpha)
{
	if(self.shape.empty())
		return;
	
	if(self.shape.size() > 1 && self.shape[1] > 1)
	{
		#pragma omp parallel for
		for(int i = 0; i < self.shape[0]; i++)
		{
			mulAdd(self[i], tgt[i], alpha);
		}

		return;
	}

	const void* src = self.data(self.offset);
	void* dest = tgt.data(tgt.offset);
	size_t stride = self.strides.back();

	switch(dtype)
	{
		case FLOAT32: {
			return cblas_saxpy(self.size(), alpha, reinterpret_cast<const float*>(src),
								stride/sizeof(float), reinterpret_cast<float*>(dest), stride/sizeof(float));
		}
		case FLOAT64: {
			return cblas_daxpy(self.size(), alpha, reinterpret_cast<const double*>(src), stride/sizeof(double), 
								reinterpret_cast<double*>(dest), stride/sizeof(double));
		}
		default: ThrowError("Unsupported dtype!");
	}
}

void BlasTensor::sub(const Tensor& self, const Tensor& b, double alpha)
{

}

template<typename T>
static void TensorDiv(
	size_t count,
	const T* a, size_t strideA,
	const T* b, size_t strideB,
	T* c, size_t strideC)
{
	for(size_t i = 0; i < count; i++)
	{
		c[i * strideC] = a[i * strideA] / b[i * strideB];
	}
}

void BlasTensor::div(const Tensor& self, const Tensor& b, Tensor tgt)
{
	if(self.shape.empty())
		return;
	
	if(self.shape.size() > 1 && self.shape[1] > 1)
	{
		#pragma omp parallel for
		for(int i = 0; i < self.shape[0]; i++)
		{
			div(self[i], b[i], tgt[i]);
		}

		return;
	}

	const void* src = self.data(self.offset);
	const void* bdata = b.data(b.offset);
	void* dest = tgt.data(tgt.offset);

	switch(dtype)
	{
		case FLOAT32: {
			return TensorDiv<float>(
						self.size(),
						reinterpret_cast<const float*>(src), self.strides.back()/sizeof(float),
						reinterpret_cast<const float*>(bdata), b.strides.back()/sizeof(float),
						reinterpret_cast<float*>(dest), tgt.strides.back()/sizeof(float));
		}
		case FLOAT64: {
			return TensorDiv<double>(
						self.size(),
						reinterpret_cast<const double*>(src), self.strides.back()/sizeof(double),
						reinterpret_cast<const double*>(bdata), b.strides.back()/sizeof(double),
						reinterpret_cast<double*>(dest), tgt.strides.back()/sizeof(double));
		}
		default: ThrowError("Unsupported dtype!");
	}
}

void BlasTensor::matmul(const Tensor& self, Tensor b, Tensor tgt)
{
	AssertExcept(self.getBackend() != tgt.getBackend(), "Cannot multiply matrices in-place!");
	if(self.shape.empty())
		return;
	
	if(self.shape.size() > 2)
	{
		//#pragma omp parallel for
		for(int i = 0; i < self.shape[0]; i++)
		{
			matmul(self[i], b[i], tgt[i]);
		}

		return;
	}

	const void* adata = self.data(self.offset);
	const void* bdata = b.data(b.offset);
	void* dest = tgt.data(tgt.offset);

	AssertExcept(self.shape[1] == b.shape[0], "A");
	//AssertExcept(self.shape[0] == tgt.shape[0] && tgt.shape[1] == b.shape[1], "B");

	auto m = tgt.shape[0];
	auto k = self.shape[1];
	auto n = b.shape[1];

	switch(dtype)
	{
		case FLOAT32: {
			cblas_sgemm(
				CblasRowMajor, CblasNoTrans, CblasNoTrans,
				m, n, k, 1.0,
				(const float*) adata, k,
				(const float*) bdata, n,
				1.0,
				(float*) dest, n
			);

		} break;
		case FLOAT64: {
			cblas_dgemm(
				CblasRowMajor, CblasNoTrans, CblasNoTrans,
				m, n, k, 1.0,
				(const double*) adata, k,
				(const double*) bdata, n,
				1.0,
				(double*) dest, n
			);
		} break;
		default: ThrowError("Unsupported dtype!");
	}
}

#include "Pool2D.hpp"

void BlasTensor::pool2d(
			const Tensor& self, Tensor tgt,
			POOLING_MODE mode,
			const UVec2& kernelSize,
			const UVec2& stride,
			const UVec2& padding,
			const UVec2& dilation)
{
	AssertExcept(self.getBackend() != tgt.getBackend(), "Cannot apply a pooling operation in-place!");
	if(self.shape.empty())
		return;
	
	if(self.shape.size() > 2)
	{
		//#pragma omp parallel for
		for(int i = 0; i < self.shape[0]; i++)
		{
			pool2d(self[i], tgt[i], mode, kernelSize, stride, padding, dilation);
		}

		return;
	}

	const void* adata = self.data(self.offset);
	void* dest = tgt.data(tgt.offset);

	const auto maxFunctor = [](const auto a, const auto b, int ksize) { return std::max(a,b); };
	const auto meanFunctor = [](const auto a, const auto b, int ksize) { return a + b/ksize; };

	switch(dtype)
	{
		case FLOAT32: {
			Pool2D<float>(
				(const float*) adata, (float*) dest,
				{self.shape[0], self.shape[1]},
				{tgt.shape[0], tgt.shape[1]},
				{self.strides[0], self.strides[1]},
				{tgt.strides[0], tgt.strides[1]},
				kernelSize, stride, padding, dilation,
				(mode == MEAN_POOLING ?
					std::function<float(const float, const float, int)>(meanFunctor):
					std::function<float(const float, const float, int)>(maxFunctor))
			);
		} break;
		case FLOAT64: {
			Pool2D<double>(
				(const double*) adata, (double*) dest,
				{self.shape[0], self.shape[1]},
				{tgt.shape[0], tgt.shape[1]},
				{self.strides[0], self.strides[1]},
				{tgt.strides[0], tgt.strides[1]},
				kernelSize, stride, padding, dilation,
				(mode == MEAN_POOLING ?
					std::function<double(const double, const double, int)>(meanFunctor):
					std::function<double(const double, const double, int)>(maxFunctor))
			);
		} break;
		default: ThrowError("Unsupported dtype!");
	}
}
