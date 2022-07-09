/* 
 * This file is part of Funzel.
 * Copyright (c) 2022 Yannick Pflanzer.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
#pragma once

#include "Funzel.hpp"
#include "Vector.hpp"

#include <memory>
#include <vector>
#include <random>

namespace funzel
{

enum DTYPE
{
	INT16 = 0,
	INT32,
	INT64,
	UINT16,
	UINT32,
	UINT64,
	FLOAT32,
	FLOAT64,
	BYTE,
	UBYTE,
	NONE,
	DTYPE_MAX
};

/**
 * @brief Determines the size of any DTYPE in bytes.
 * 
 * This function is comparable to the builtin 'sizeof'.
 * 
 * @param dtype The DTYPE to determine the size of.
 * @return The size of the type in bytes.
 */
inline constexpr size_t dtypeSizeof(const DTYPE dtype)
{
	switch(dtype)
	{
		case INT16:
		case UINT16: return 2;

		case FLOAT32:
		case UINT32:
		case INT32: return 4;

		case FLOAT64:
		case UINT64:
		case INT64: return 8;
	
		case BYTE:
		case UBYTE:
			return 1;

		case NONE:
		default:
			return 0;
	}
}

inline std::string dtypeToNativeString(const DTYPE dtype)
{
	switch(dtype)
	{
		case UINT16: return "ushort";
		case INT16: return "short";

		case FLOAT32: return "float";
		case UINT32: return "uint";
		case INT32: return "int";

		case FLOAT64: return "double";
		case UINT64: return "ulong";
		case INT64: return "long";
	
		case BYTE: return "char";
		case UBYTE: return "uchar";

		default:
		case NONE: return "void";
	}
}

#ifndef SWIG
template<typename T>
DTYPE dtype()
{
	if constexpr(std::is_same_v<T, double>) return FLOAT64;
	else if constexpr(std::is_same_v<T, float>) return FLOAT32;
	else if constexpr(std::is_same_v<T, int64_t>) return INT64;
	else if constexpr(std::is_same_v<T, uint64_t>) return UINT64;
	else if constexpr(std::is_same_v<T, int>) return INT32;
	else if constexpr(std::is_same_v<T, unsigned int>) return UINT32;
	else if constexpr(std::is_same_v<T, short>) return INT16;
	else if constexpr(std::is_same_v<T, unsigned short>) return UINT16;
	else if constexpr(std::is_same_v<T, char>) return BYTE;
	else if constexpr(std::is_same_v<T, unsigned char>) return UBYTE;
	return NONE;
}

template<typename T>
DTYPE dtypeOf(const T&) { return dtype<T>(); }
#endif

typedef std::vector<size_t> Shape;
typedef std::vector<size_t> Index;

enum TensorFlags
{
	C_CONTIGUOUS = 0x1,
	WRITEABLE = 0x2
};

class BackendTensor;

/**
 * @brief 
 * 
 */
class FUNZEL_API Tensor
{
public:
	static Tensor ones(size_t count, DTYPE dtype = FLOAT32, const std::string& backend = EmptyStr);
	static Tensor zeros(size_t count, DTYPE dtype = FLOAT32, const std::string& backend = EmptyStr);
	
	static Tensor empty(const Shape& shape, const std::shared_ptr<char>& data, DTYPE dtype = FLOAT32, const std::string& backend = EmptyStr);
	static Tensor empty(const Shape& shape, const void* data, DTYPE dtype = FLOAT32, const std::string& backend = EmptyStr);
	static Tensor empty(const Shape& shape, DTYPE dtype = FLOAT32, const std::string& backend = EmptyStr);
	static Tensor empty_like(const Tensor& t);
	
	static Tensor ones(const Shape& shape, DTYPE dtype = FLOAT32, const std::string& backend = EmptyStr);
	static Tensor ones_like(const Tensor& t);

	static Tensor zeros(const Shape& shape, DTYPE dtype = FLOAT32, const std::string& backend = EmptyStr);
	static Tensor zeros_like(const Tensor& t);

	Tensor() = default;
	explicit Tensor(const Shape& shape, const float data[], unsigned long long sz, const std::string& device = EmptyStr);
	explicit Tensor(const Shape& shape, const double data[], unsigned long long sz, const std::string& device = EmptyStr);

	//explicit Tensor(const Shape& shape, const std::vector<double>& data, const std::string& device = EmptyStr);

	explicit Tensor(const Shape& shape, std::initializer_list<float> data, const std::string& device = EmptyStr);
	explicit Tensor(const Shape& shape, std::initializer_list<double> data, const std::string& device = EmptyStr);

	BackendTensor* getBackend() { return m_backend.get(); }
	const BackendTensor* getBackend() const { return m_backend.get(); }

#ifndef SWIG
	BackendTensor* operator->() { return m_backend.get(); }
	BackendTensor* operator->() const { return m_backend.get(); }

	template<typename T>
	T* getBackendAs() { return dynamic_cast<T*>(m_backend.get()); }

	template<typename T>
	T* getBackendAs() const { return dynamic_cast<T*>(m_backend.get()); }
#endif

	std::string toString() const;

	bool empty() const { return m_backend == nullptr; }
	size_t size() const;
	Tensor get(const Index& idx) const;
	Tensor get(size_t idx) const;
	void* data(size_t offset = 0);
	const void* data(size_t offset = 0) const;
	
	template<typename T> 
	const T& dataAs(size_t offset) const
	{
		return *reinterpret_cast<const T*>(data(offset*sizeof(T)));
	}

	template<typename T> 
	T& dataAs(size_t offset)
	{
		return *reinterpret_cast<T*>(data(offset*sizeof(T)));
	}

	Tensor astype(DTYPE type) const;

	template<typename T>
	Tensor astype() const
	{
		return astype(funzel::dtype<T>());
	}

	Tensor operator[](size_t idx) const { return get(idx); }
	Tensor operator[](const Index& idx) const { return get(idx); }

	template<typename T>
	void set(T t)
	{
		switch(dtype)
		{
			case FLOAT32: ritem<float>() = t; break;
			case UINT32: ritem<uint32_t>() = t; break;
			case INT32: ritem<int32_t>() = t; break;
			case FLOAT64: ritem<double>() = t; break;
			case UINT64: ritem<uint64_t>() = t; break;
			case INT64: ritem<int64_t>() = t; break;
			case BYTE: ritem<char>() = t; break;
			case UBYTE: ritem<unsigned char>() = t; break;
			
			default:
			case NONE: break;
		}
	}

	void set(const Tensor& t);
	
	template<typename T>
	Tensor& operator=(T v) { set(v); return *this; }

	template<typename T>
	T item() const
	{
		AssertExcept(shape.empty() || (shape.size() == 1 && shape[0] == 1), "Can only take the item of a one-element tensor!");
		const void* data = this->data(offset);

		switch(dtype)
		{
			case FLOAT32: return *reinterpret_cast<const float*>(data); break;
			case UINT32: return *reinterpret_cast<const uint32_t*>(data); break;
			case INT32: return *reinterpret_cast<const int32_t*>(data); break;
			case FLOAT64: return *reinterpret_cast<const double*>(data); break;
			case UINT64: return *reinterpret_cast<const uint64_t*>(data); break;
			case INT64: return *reinterpret_cast<const int64_t*>(data); break;
			case BYTE: return *reinterpret_cast<const char*>(data); break;

			default:
			case UBYTE: return *reinterpret_cast<const unsigned char*>(data); break;
		}
	}

	template<typename T>
	T& ritem()
	{
		AssertExcept(shape.empty() || (shape.size() == 1 && shape[0] == 1), "Can only take the item of a one-element tensor!");
		void* data = this->data(offset);
		return *reinterpret_cast<T*>(data);
	}

	void trimDimensions();

	// Operations
	Tensor reshape(const Shape& shape);
	void reshape_(const Shape& shape);

	Tensor cpu() const;
	Tensor to(const std::string& device) const;

	Tensor clone() const;
	Tensor unravel() const;
	Tensor transpose() const;

	Tensor operator+(const Tensor& t) const { return add(t); }
	Tensor operator-(const Tensor& t) const { return sub(t); }
	Tensor operator*(const Tensor& t) const { return matmul(t); }
	
	Tensor operator+(double t) const { return add(t); }
	Tensor operator*(double t) const { return mul(t); }
	Tensor operator/(double t) const { return mul(1.0/t); }
	Tensor operator/(const Tensor& t) const { return div(t); }

	Tensor operator-() const { return mul(-1.0); }

	Tensor& fill(double value);

	Tensor mul(double alpha) const;
	Tensor& mul_(double alpha);

	Tensor div(const Tensor& b) const;
	Tensor& div_(const Tensor& b);

	Tensor matmul(const Tensor& b) const;

	Tensor& add_(const Tensor& b, double alpha = 1.0);
	Tensor add(const Tensor& b, double alpha = 1.0) const;

	Tensor add(double alpha) const;
	Tensor& add_(double alpha);

	Tensor& sub_(const Tensor& b, double alpha = 1.0);
	Tensor sub(const Tensor& b, double alpha = 1.0) const;

	/**
	 * @brief 
	 * \f$ |x| \f$
	 * @return Tensor 
	 */
	Tensor abs() const;
	Tensor& abs_();

	/**
	 * @brief 
	 * \f$ e^x \f$
	 * @return Tensor 
	 */
	Tensor exp() const;
	Tensor& exp_();

	Tensor sqrt() const;
	Tensor& sqrt_();

	Tensor sin() const;
	Tensor& sin_();

	Tensor cos() const;
	Tensor& cos_();

	Tensor tan() const;
	Tensor& tan_();

	Tensor tanh() const;
	Tensor& tanh_();

	Tensor sigmoid() const;
	Tensor& sigmoid_();

	double sum();

	bool isContiguous() const { return flags & C_CONTIGUOUS; }

	DTYPE dtype;
	Shape shape, strides;
	size_t offset = 0;
	uint32_t flags = C_CONTIGUOUS;
	std::string device;

private:
	std::shared_ptr<BackendTensor> m_backend = nullptr;
};

inline Tensor operator+(double v, const Tensor& t) { return t.add(v); }
inline Tensor operator*(double v, const Tensor& t) { return t.mul(v); }
inline Tensor operator/(double v, const Tensor& t) { return Tensor::empty_like(t).fill(v).div_(t); }

enum POOLING_MODE
{
	MEAN_POOLING,
	MAX_POOLING
};

class FUNZEL_API BackendTensor
{
public:
	virtual ~BackendTensor() {}

	// Placeholder to be overridden!
	static void initializeBackend() {}

	virtual void empty(std::shared_ptr<char> buffer, size_t sz, const Shape& shape, DTYPE dtype = FLOAT32) = 0;
	virtual void empty(const void* buffer, size_t sz, const Shape& shape, DTYPE dtype = FLOAT32) = 0;

	virtual void* data(size_t offset = 0) = 0;
	virtual std::shared_ptr<char> buffer() = 0;
	virtual std::shared_ptr<BackendTensor> clone() const = 0;
	virtual const char* backendName() const = 0;

	virtual void fill(const Tensor& self, double scalar) = 0;
	virtual void mulAdd(const Tensor& self, Tensor tgt, double alpha) { ThrowError("Operation is not supported!"); }
	virtual void matmul(const Tensor& self, Tensor b, Tensor tgt) { ThrowError("Operation is not supported!"); }
	virtual void div(const Tensor& self, const Tensor& b, Tensor tgt) { ThrowError("Operation is not supported!"); }

	virtual void sub(const Tensor& self, const Tensor& b, double alpha = 1.0) { ThrowError("Operation is not supported!"); }
	virtual void abs(const Tensor& self, Tensor tgt) { ThrowError("Operation is not supported!"); }
	virtual void exp(const Tensor& self, Tensor tgt) { ThrowError("Operation is not supported!"); }
	virtual void sqrt(const Tensor& self, Tensor tgt) { ThrowError("Operation is not supported!"); }
	virtual void sin(const Tensor& self, Tensor tgt) { ThrowError("Operation is not supported!"); }
	virtual void cos(const Tensor& self, Tensor tgt) { ThrowError("Operation is not supported!"); }
	virtual void tan(const Tensor& self, Tensor tgt) { ThrowError("Operation is not supported!"); }
	virtual void tanh(const Tensor& self, Tensor tgt) { ThrowError("Operation is not supported!"); }
	virtual double sum(const Tensor& self) { ThrowError("Operation is not supported!"); return 0; }

	virtual void pool2d(
			const Tensor& self, Tensor tgt,
			POOLING_MODE mode,
			const UVec2& kernelSize,
			const UVec2& stride,
			const UVec2& padding,
			const UVec2& dilation) { ThrowError("Operation is not supported!"); }

	virtual void conv2d(
			const Tensor& self, Tensor tgt,
			const Tensor& kernel,
			const UVec2& stride,
			const UVec2& padding,
			const UVec2& dilation) { ThrowError("Operation is not supported!"); }

	virtual void set(Tensor& self, const Tensor& src) { ThrowError("Operation is not supported!"); }

	// With default implementation
	virtual void sigmoid(const Tensor& self, Tensor& tgt);
	
	DTYPE dtype;
	size_t size;

private:
};

inline size_t size(const Shape& shape)
{
	if(shape.empty())
		return 0;
	
	size_t sz = 1;
	for(const auto dim : shape)
		sz *= dim;

	return sz;
}

FUNZEL_API Tensor linspace(double start, double stop, size_t num, bool endPoint = true, DTYPE dtype = FLOAT32);
FUNZEL_API Tensor linspace(const Tensor& start, const Tensor& stop, size_t num, bool endPoint = true, DTYPE dtype = FLOAT32);
FUNZEL_API Tensor logspace(const Tensor& start, const Tensor& stop, size_t num, bool endPoint = true, double base = 10.0, DTYPE dtype = FLOAT32);
FUNZEL_API Tensor arange(double start, double stop, double step, DTYPE dtype = FLOAT32);

struct IRandomGenerator
{
	virtual ~IRandomGenerator() {}
	virtual double get() = 0;
};

template<typename Dist>
struct RandomGenerator : public IRandomGenerator
{
	RandomGenerator() = default;
	RandomGenerator(const Dist& d, uint64_t seed = std::mt19937::default_seed):
		distribution(d),
		gen(seed) {}
	
	double get() override
	{
		return distribution(gen);
	}
	
	Dist distribution;
	std::mt19937 gen;
};

FUNZEL_API Tensor& randn(Tensor& out, IRandomGenerator& generator);
FUNZEL_API Tensor& randn(Tensor& out);

#ifndef SWIG
FUNZEL_API std::ostream& operator<<(std::ostream& out, const Tensor& s);
FUNZEL_API std::ostream& operator<<(std::ostream& out, const Shape& s);
#endif

}
