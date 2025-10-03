#pragma once

#include "BlasTensor.hpp"
#include "funzel/Funzel.hpp"

#include <lapacke.h>
#include <cblas.h>
#include <spdlog/spdlog.h>

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

	~BlasTensorImpl() = default;

	void fill(const Tensor& self, double scalar) override
	{
		TensorOp(self, self, [scalar](const auto& v) -> T { return T(scalar); });
	}

	void unravel(const Tensor& self, Tensor tgt) override
	{
		TensorOp(self, tgt, [](const auto& v) { return v; });
	}

	void set(Tensor& self, const Tensor& src) override
	{
		unravel(src, self);
	}

	template<typename AccumT>
	static AccumT sum(const T* FUNZEL_RESTRICT data, size_t n, size_t stride = 1) FUNZEL_NOINLINE;

	void sum(
		const Tensor& self,
		Tensor& tgt,
		const small_vector<int>& axis,
		DTYPE dtype,
		bool keepdims) override
	{
		Reduce<int>(self, axis, dtype, tgt, keepdims, std::optional<int>{0}, [](const Tensor& t1, Tensor& t2) {
			t2.add_(t1);
		});
	}

	template<typename Fn>
	inline void TensorOpInner(const Tensor& self, Tensor& tgt, Fn op)
	{
		using TargetT = std::invoke_result_t<Fn, T>;

		const T* src = reinterpret_cast<const T*>(self.data(self.offset));
		TargetT* dest = reinterpret_cast<TargetT*>(tgt.data(tgt.offset));
		
		const int64_t srcStride = self.strides[0] / int(sizeof(T));
		const int64_t tgtStride = tgt.strides[0] / int(sizeof(TargetT));

		const int64_t srcOffset = srcStride < 0 ? (self.shape[0] - 1) * std::abs(srcStride) : 0;
		const int64_t tgtOffset = tgtStride < 0 ? (tgt.shape[0] - 1) * std::abs(tgtStride) : 0;

		//spdlog::info("SIZE: {} vs {} self.offset = {} offset: {} stride: {}", self.size(), self->size/sizeof(T), self.offset, srcOffset, srcStride);

		//#pragma omp parallel for if(self.shape[0] > 4096)
		for(int64_t x = 0; x < self.shape[0]; x++)
		{
			//tgt.dataAs<TargetT>(tgtOffset + x*tgtStride) = op(self.dataAs<T>(srcOffset + x*srcStride));
			//spdlog::info("OP {} -> {}", srcOffset + x*srcStride, tgtOffset + x*tgtStride);
			dest[tgtOffset + x * tgtStride] = op(src[srcOffset + x * srcStride]);
		}
	}

	template<typename Fn>
	inline void TensorOp(const Tensor& self, Tensor tgt, Fn op)
	{
		if(self.shape.size() > 1)
		{
			//#pragma omp parallel for if(self.shape[0] > 4096)
			for(int64_t i = 0; i < self.shape[0]; i++)
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

	template<typename TargetT>
	inline TargetT convertValue(const TargetT& v)
	{
		if constexpr (std::is_same_v<TargetT, T>)
		{
			return v;
		}
		else
		{
			return static_cast<TargetT>(v);
		}
	}

	void astype(const Tensor& self, Tensor tgt, DTYPE targetType) override
	{
		AssertExcept(tgt->dtype == targetType, "Target tensor does not match the given target DTYPE!");
		switch(targetType)
		{
			case DFLOAT32: TensorOp(self, tgt, [this](const auto& v) { return convertValue<float>(v); }); break;
			case DUINT32: TensorOp(self, tgt, [this](const auto& v) { return convertValue<uint32_t>(v); }); break;
			case DINT32: TensorOp(self, tgt, [this](const auto& v) { return convertValue<int32_t>(v); }); break;
			case DFLOAT64: TensorOp(self, tgt, [this](const auto& v) { return convertValue<double>(v); }); break;
			case DUINT64: TensorOp(self, tgt, [this](const auto& v) { return convertValue<uint64_t>(v); }); break;
			case DINT64: TensorOp(self, tgt, [this](const auto& v) { return convertValue<int64_t>(v); }); break;
			case DINT8: TensorOp(self, tgt, [this](const auto& v) { return convertValue<int8_t>(v); }); break;
			case DUINT8: TensorOp(self, tgt, [this](const auto& v) { return convertValue<uint8_t>(v); }); break;
			case DINT16: TensorOp(self, tgt, [this](const auto& v) { return convertValue<int16_t>(v); }); break;
			case DUINT16: TensorOp(self, tgt, [this](const auto& v) { return convertValue<uint16_t>(v); }); break;
			default: throw std::invalid_argument("Unsupported DTYPE given: " + std::to_string(targetType));
		}
	}

	void mean(const Tensor& self, Tensor& tgt, const small_vector<int>& axis, DTYPE dtype, bool keepdims) override
	{
		sum(self, tgt, axis, dtype, keepdims);

		// Calculate normalizing factor
		small_vector<double> norm(tgt.shape.size());

		if(axis.empty())
			norm[0] = 1.0/self.size();
		else
		{

			// Axes needs to be sorted descending!
			auto sortedAx = axis;
			std::sort(sortedAx.begin(), sortedAx.end(), std::greater<int>());

			for(int i = 0; i < sortedAx.size(); i++)
			{
				norm[i] = 1.0/self.shape[sortedAx[i]];
			}
		}
		
		auto normTensor = Tensor::empty({tgt.shape.size()}, norm.data(), DTYPE::DFLOAT64, self.device);
		tgt.mul_(normTensor.astype(tgt.dtype));
	}

	void add(const Tensor& a, const Tensor& b, Tensor tgt) override
	{
		Broadcast<0>(a, b, tgt,
			[](const auto& a, const auto& b) { return a; },
			[](const Tensor& a, Tensor b, Tensor tgt) {
				funzel::ApplyStrided(a, b, tgt, [](const auto& a, const auto& b, auto tgt) {
					const T* adata = reinterpret_cast<const T*>(a.data(a.offset));
					const T* bdata = reinterpret_cast<const T*>(b.data(b.offset));
					T* dest = reinterpret_cast<T*>(tgt.data(tgt.offset));

					const size_t tgtStride = tgt.strides[0] / sizeof(T);
					const size_t aStride = a.strides[0] / sizeof(T);
					const size_t bStride = b.strides[0] / sizeof(T);

					for(size_t i = 0; i < a.size(); i++)
					{
						dest[i*tgtStride] = adata[i*aStride] + bdata[i*bStride];
					}
				});
		});
	}

	static void axpy(
		const T* FUNZEL_RESTRICT a,
		const T* FUNZEL_RESTRICT b,
		T* FUNZEL_RESTRICT dest,
		T alpha,
		
		size_t n,
		size_t astride = 1,
		size_t bstride = 1,
		size_t deststride = 1) FUNZEL_NOINLINE
	{
		for(size_t i = 0; i < n; i++)
		{
			dest[i*deststride] = a[i*astride] + b[i*bstride]*alpha;
		}
	}

	void mulAdd(const Tensor& self, Tensor tgt, double alpha)
	{
		#if 0
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
		#endif

		//funzel::ApplyStrided(self, tgt, [alpha](const auto& self, auto tgt) {
		funzel::Apply(self, tgt, tgt, 1, [](const auto& self, const auto& b, auto tgt, double alpha) {
			const T* src = reinterpret_cast<const T*>(self.data(self.offset));
			T* dest = reinterpret_cast<T*>(tgt.data(tgt.offset));
			size_t destStride = tgt.strides.back();
			size_t stride = self.strides.back();

			axpy(src, src, dest, alpha, self.size(), stride/sizeof(T), stride/sizeof(T), destStride/sizeof(T));
			// std::cout << self << " " << tgt << std::endl;

#if 0
			if constexpr (std::is_same_v<T, float>)
			{
				cblas_saxpy(self.size(), alpha, reinterpret_cast<const float*>(src),
										stride/sizeof(float), reinterpret_cast<float*>(dest), destStride/sizeof(float));

				std::cout << self << " " << tgt << std::endl;
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
#endif
		}, alpha);
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

	template<typename V>
	static void TensorMul(
		size_t count,
		const V* a, size_t strideA,
		const V* b, size_t strideB,
		V* c, size_t strideC)
	{
		for(size_t i = 0; i < count; i++)
		{
			c[i * strideC] = a[i * strideA] * b[i * strideB];
		}
	}

	template<typename V>
	static void ScalarMul(
		size_t count,
		const V* a, size_t strideA,
		V scalar,
		V* c, size_t strideC)
	{
		for(size_t i = 0; i < count; i++)
		{
			c[i * strideC] = a[i * strideA] * scalar;
		}
	}

	void mul(const Tensor& self, const Tensor& b, Tensor tgt)
	{
		funzel::ApplyStrided(self, b, tgt, [](const auto& self, const auto& b, auto tgt) {
			const void* src = self.data(self.offset);
			const T* bdata = reinterpret_cast<const T*>(b.data(b.offset));
			void* dest = tgt.data(tgt.offset);

			if(b.size() == 1) // Scalar!
			{
				ScalarMul<T>(
							self.size(),
							reinterpret_cast<const T*>(src), self.strides.back()/sizeof(T),
							*bdata,
							reinterpret_cast<T*>(dest), tgt.strides.back()/sizeof(T));
			}
			else
			{
				TensorMul<T>(
							self.size(),
							reinterpret_cast<const T*>(src), self.strides.back()/sizeof(T),
							reinterpret_cast<const T*>(bdata), b.strides.back()/sizeof(T),
							reinterpret_cast<T*>(dest), tgt.strides.back()/sizeof(T));
			}
		});
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
	void svd(const Tensor& self, Tensor U, Tensor S, Tensor V, bool fullMatrices = true) override;
};

}
}

#include "BlasTensorImpl.inl"
#include "BlasTensorImpl_NN.inl"
#include "BlasTensorImpl_CV.inl"
#include "BlasTensorImpl_Linalg.inl"
