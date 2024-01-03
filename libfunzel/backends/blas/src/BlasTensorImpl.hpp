#pragma once

#include "BlasTensor.hpp"

#include <lapacke.h>
#include <cblas.h>

namespace funzel
{
namespace blas
{

enum SIMD_TYPE
{
	NONE,
	AVX2,
	AVX512,
	NEON
};

template<typename T, SIMD_TYPE SimdType = NONE>
class EXPORT BlasTensorImpl: public BlasTensor
{
public:
	BlasTensorImpl()
	{
		dtype = funzel::dtype<T>();
	}

	void fill(const Tensor& self, double scalar) override
	{
		const auto sz = self.size();
		auto* d = reinterpret_cast<T*>(data(self.offset));
		
		// TODO Make configurable!
		#pragma omp parallel for if(sz > 4096)
		for(int64_t i = 0; i < sz; i++)
		{
			d[i] = scalar;
		}
	}

	template<typename AccumT>
	AccumT sum(const T* FUNZEL_RESTRICT data, size_t n, size_t stride = 1) FUNZEL_NOINLINE;

	double sum(const Tensor& self) override
	{
		if(self.shape.empty())
			return 0;
		
		if(self.shape.size() > 1)
		{
			double s = 0;
			for(int i = 0; i < self.shape[0]; i++)
			{
				s += sum(self[i]);
			}

			return s;
		}

		return sum<double>((T*) data(self.offset), self.size(), self.strides.back() / sizeof(T));
#if 0
		void* data = this->data(self.offset);
		size_t stride = self.strides.back();

		if constexpr (std::is_same_v<T, float>)
		{
			const float one = 1.0f;
			return cblas_sdot(self.size(), reinterpret_cast<float*>(data), stride/sizeof(float), &one, 0);
		}
		else if constexpr (std::is_same_v<T, double>)
		{
			const double one = 1.0;
			return cblas_ddot(self.size(), reinterpret_cast<double*>(data), stride/sizeof(double), &one, 0);
		}
		else
		{
			throw std::runtime_error("Unsupported type for operation!");
		}
#endif
		return 0;
	}

	template<typename Fn>
	inline void TensorOpInner(const Tensor& self, Tensor& tgt, Fn op)
	{
		for(int64_t x = 0; x < self.shape[0]; x++)
		{
			T& v = self[x].ritem<T>();
			tgt[x].ritem<T>() = op(v);
		}
	}

	template<typename Fn>
	inline void TensorOp(const Tensor& self, Tensor tgt, Fn op)
	{
		if(self.shape.size() > 1)
		{
			for(int i = 0; i < self.shape[0]; i++)
				TensorOp(self[i], tgt[i], op);

			return;
		}

		TensorOpInner(self, tgt, op);
	}

	void abs(const Tensor& self, Tensor tgt)
	{
		// Abs on unsigned values is a no-op!
		if constexpr (!std::is_unsigned_v<T>)
		{
			TensorOp(self, tgt, [](const auto& v) { return std::abs(v); });
		}
	}

	void exp(const Tensor& self, Tensor tgt)
	{
		TensorOp(self, tgt, [](const auto& v) { return std::exp(v); });
	}

	void sqrt(const Tensor& self, Tensor tgt)
	{
		TensorOp(self, tgt, [](const auto& v) { return std::sqrt(v); });
	}

	void sin(const Tensor& self, Tensor tgt)
	{
		TensorOp(self, tgt, [](const auto& v) { return std::sin(v); });
	}

	void cos(const Tensor& self, Tensor tgt)
	{
		TensorOp(self, tgt, [](const auto& v) { return std::cos(v); });
	}

	void tan(const Tensor& self, Tensor tgt)
	{
		TensorOp(self, tgt, [](const auto& v) { return std::tan(v); });
	}

	void tanh(const Tensor& self, Tensor tgt)
	{
		TensorOp(self, tgt, [](const auto& v) { return std::tanh(v); });
	}

	template<typename Result, typename Input>
	Result meanFlattened(Tensor& self) const
	{
		const auto size = self.size();
		Result accum(0);

		#if 0
		if constexpr (std::is_same_v<Input, float>)
			accum = cblas_sasum(size, self.data(), self.strides.back());
		else if constexpr (std::is_same_v<Input, double>)
			accum = cblas_dasum(size, self.data(), self.strides.back());
		else if constexpr (std::is_same_v<Input, float>)
			accum = cblas_sasum(size, self.data(), self.strides.back());
		else // Use our own, (unoptimized) version
		#endif
		{
			//for(size_t i = 0; i < size; i++)
			//	accum += self.dataAs<Input>(i);
		}

		return sum(self) / size;
	}

	void mean(const Tensor& self, Tensor& tgt, const small_vector<int>& axis, DTYPE dtype, bool keepdims) override
	{
		// Use flattened version
		if(axis.empty() || axis[0] == -1)
		{
			tgt = sum(self) / self.size();
			//auto v = meanFlattened<>(self);
		}
	}

	void mulAdd(const Tensor& self, Tensor tgt, double alpha)
	{
		if(self.shape.empty())
			return;
		
		if(self.shape.size() > 1 && self.shape[1] > 1)
		{
			//#pragma omp parallel for
			for(int i = 0; i < self.shape[0]; i++)
			{
				mulAdd(self[i], tgt[i], alpha);
			}

			return;
		}

		const void* src = self.data(self.offset);
		void* dest = tgt.data(tgt.offset);
		size_t destStride = tgt.strides.back();
		size_t stride = self.strides.back();

		if constexpr (std::is_same_v<T, float>)
		{
			cblas_saxpy(self.size(), alpha, reinterpret_cast<const float*>(src),
									stride/sizeof(float), reinterpret_cast<float*>(dest), destStride/sizeof(float));
		}
		else if constexpr (std::is_same_v<T, double>)
		{
			cblas_daxpy(self.size(), alpha, reinterpret_cast<const double*>(src), stride/sizeof(double), 
									reinterpret_cast<double*>(dest), destStride/sizeof(double));
		}
		else
		{
			throw std::runtime_error("Unsupported type for operation!");
		}
	}

	void mul(Tensor self, double alpha)
	{
		if(self.shape.empty())
			return;
		
		if(self.shape.size() > 1 && self.shape[1] > 1)
		{
			//#pragma omp parallel for
			for(int i = 0; i < self.shape[0]; i++)
			{
				mul(self[i], alpha);
			}

			return;
		}

		void* src = self.data(self.offset);
		size_t stride = self.strides.back();

		if constexpr (std::is_same_v<T, float>)
		{
			cblas_sscal(self.size(), alpha, reinterpret_cast<float*>(src), stride/sizeof(float));
		}
		else if constexpr (std::is_same_v<T, double>)
		{
			cblas_sscal(self.size(), alpha, reinterpret_cast<float*>(src), stride/sizeof(double));
		}
		else
		{
			throw std::runtime_error("Unsupported type for operation!");
		}
	}

	template<typename V>
	static void TensorDiv(
		size_t count,
		const V* a, size_t strideA,
		const V* b, size_t strideB,
		V* c, size_t strideC)
	{
		for(size_t i = 0; i < count; i++)
		{
			c[i * strideC] = a[i * strideA] / b[i * strideB];
		}
	}

	void div(const Tensor& self, const Tensor& b, Tensor tgt)
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

		return TensorDiv<T>(
							self.size(),
							reinterpret_cast<const T*>(src), self.strides.back()/sizeof(T),
							reinterpret_cast<const T*>(bdata), b.strides.back()/sizeof(T),
							reinterpret_cast<T*>(dest), tgt.strides.back()/sizeof(T));
	}

	void matmul(const Tensor& self, Tensor b, Tensor tgt)
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

		if constexpr (std::is_same_v<T, float>)
		{
			cblas_sgemm(
					CblasRowMajor, CblasNoTrans, CblasNoTrans,
					m, n, k, 1.0,
					(const float*) adata, k,
					(const float*) bdata, n,
					1.0,
					(float*) dest, n
				);
		}
		else if constexpr (std::is_same_v<T, double>)
		{
			cblas_dgemm(
					CblasRowMajor, CblasNoTrans, CblasNoTrans,
					m, n, k, 1.0,
					(const double*) adata, k,
					(const double*) bdata, n,
					1.0,
					(double*) dest, n
				);
		}
		else
		{
			throw std::runtime_error("Unsupported type for operation!");
		}
	}

	// NNBackendTensor
	void pool2d(
			const Tensor& self, Tensor tgt,
			POOLING_MODE mode,
			const UVec2& kernelSize,
			const UVec2& stride,
			const UVec2& padding,
			const UVec2& dilation);

	void relu(const Tensor& self, Tensor& tgt, double negativeSlope);

	// CVBackendTensor
	void conv2d(
		const Tensor& self, Tensor tgt,
		const Tensor& kernel,
		const UVec2& stride,
		const UVec2& padding,
		const UVec2& dilation);

	void convertGrayscale(const Tensor& self, Tensor tgt) override;

	// LinalgBackendTensor
	void det(const Tensor& self, Tensor tgt) override;
	void inv(const Tensor& self, Tensor tgt) override;
	void trace(const Tensor& self, Tensor tgt) override;
	void svd(const Tensor& self, Tensor U, Tensor S, Tensor V) override;
};

}
}

#include "BlasTensorImpl.inl"
#include "BlasTensorImpl_NN.inl"
#include "BlasTensorImpl_CV.inl"
#include "BlasTensorImpl_Linalg.inl"
