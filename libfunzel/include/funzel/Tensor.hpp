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
#include "funzel/Type.hpp"
#include "small_vector"

#include <memory>
#include <random>
#include <functional>
#include <optional>
#include <numeric>

namespace funzel
{

typedef small_vector<size_t> Shape;
typedef small_vector<int64_t> Index;
typedef small_vector<int64_t> Strides;
//typedef std::vector<size_t> Shape;
//typedef std::vector<size_t> Index;

/**
 * @brief Represents a slice of a tensor.
 * 
 * The default values for the parameters are chosen to represent the entire tensor.
 * first = 0 (start at beginning), last = -1 (until end), step = 1 (every element)
 */
struct TensorSlice
{
	int64_t first = 0;
	int64_t last = -1;
	int64_t step = 1;
};

enum TensorFlags
{
	C_CONTIGUOUS = 0x1,
	WRITEABLE = 0x2
};

class BackendTensor;

/**
 * @brief Implements a tensor with basic operations like creation and arithmetic.
 * 
 * A Tensor wraps around a BackendTensor to abstract away device specific
 * implementation details.
 * 
 * As a convention, all methods ending with '_' will change the object
 * they were called on in-place, while all other methods will return
 * a new Tensor object which may for some operations share the same backend as the
 * original object. Use 'clone' if you need to write to the memory while preserving
 * the original data.
 * 
 * In the documentation, 'x' will be used as a signifier of 'this' in mathematical
 * equations to make them more readable.
 * 
 */
class FUNZEL_API Tensor
{
public:
	/**
	 * @brief Creates a new tensor containing all 1's
	 * 
	 * @param count The number of elements.
	 * @param dtype (optional) The type of each element.
	 * @param backend (optional) The device string.
	 * @return Tensor A new Tensor of shape (count).
	 */
	static Tensor ones(size_t count, DTYPE dtype = DFLOAT32, const std::string& backend = EmptyStr);

	/**
	 * @brief Creates a new tensor containing all 0's
	 * 
	 * @param count The number of elements.
	 * @param dtype (optional) The type of each element.
	 * @param backend (optional) The device string.
	 * @return Tensor A new Tensor of shape (count).
	 */
	static Tensor zeros(size_t count, DTYPE dtype = DFLOAT32, const std::string& backend = EmptyStr);
	
	#ifndef SWIG
	/**
	 * @brief Creates a new empty tensor.
	 * 
	 * @param shape The shape of the new tensor.
	 * @param data (optional) A pointer to data that should be used for initialization.
	 * @param dtype (optional) The type of each element.
	 * @param backend (optional) The device string.
	 * @return Tensor A new Tensor of given shape.
	 */
	static Tensor empty(const Shape& shape, const std::shared_ptr<char>& data, DTYPE dtype = DFLOAT32, const std::string& backend = EmptyStr);

	/**
	 * @brief Creates a new empty tensor.
	 * 
	 * @param shape The shape of the new tensor.
	 * @param data (optional) A pointer to data that should be used for initialization.
	 * @param dtype (optional) The type of each element.
	 * @param backend (optional) The device string.
	 * @return Tensor A new Tensor of given shape.
	 */
	static Tensor empty(const Shape& shape, const void* data, DTYPE dtype = DFLOAT32, const std::string& backend = EmptyStr);
	#endif
	
	/**
	 * @brief Creates a new empty tensor with uninitialized memory.
	 * 
	 * @param shape The shape of the new tensor.
	 * @param dtype (optional) The type of each element.
	 * @param backend (optional) The device string.
	 * @return Tensor A new Tensor of given shape.
	 */
	static Tensor empty(const Shape& shape, DTYPE dtype = DFLOAT32, const std::string& backend = EmptyStr);

	/**
	 * @brief Creates a new empty tensor with the same shape, type and device as the template tensor.
	 * 
	 * @param t The tensor to use as a template.
	 * @return Tensor A new Tensor.
	 */
	static Tensor empty_like(const Tensor& t);
	
	/**
	 * @brief Creates a new tensor of 1's.
	 * 
	 * @param shape The shape of the new tensor.
	 * @param dtype (optional) The type of each element.
	 * @param backend (optional) The device string.
	 * @return Tensor A new Tensor of given shape.
	 */
	static Tensor ones(const Shape& shape, DTYPE dtype = DFLOAT32, const std::string& backend = EmptyStr);
	
	/**
	 * @brief Creates a new tensor of 1's with the same shape, type and device as the template tensor.
	 * 
	 * @param t The tensor to use as a template.
	 * @return Tensor A new Tensor.
	 */
	static Tensor ones_like(const Tensor& t);

	/**
	 * @brief Creates a new tensor of 0's.
	 * 
	 * @param shape The shape of the new tensor.
	 * @param dtype (optional) The type of each element.
	 * @param backend (optional) The device string.
	 * @return Tensor A new Tensor of given shape.
	 */
	static Tensor zeros(const Shape& shape, DTYPE dtype = DFLOAT32, const std::string& backend = EmptyStr);

	/**
	 * @brief Creates a new tensor of 0's with the same shape, type and device as the template tensor.
	 * 
	 * @param t The tensor to use as a template.
	 * @return Tensor A new Tensor.
	 */
	static Tensor zeros_like(const Tensor& t);

	template<typename T>
	static Tensor scalar(T v, const std::string& device = EmptyStr) { return Tensor({1}, {v}, device); }

	template<typename T>
	static Tensor scalar(T v, DTYPE dtype, const std::string& device = EmptyStr) { return Tensor({1}, {v}, device).astype(dtype); }

	template<typename T>
	static Tensor scalar_like(const Tensor& t, T v) { return Tensor({1}, {v}, t.device).astype(t.dtype); }

	Tensor() = default;
	~Tensor() = default;

	Tensor(const Tensor&) = default;
	Tensor& operator=(const Tensor&) = default;

	Tensor(Tensor&& t)
	{
		*this = std::move(t);
	}
	
	Tensor& operator=(Tensor&& t)
	{
		shape = std::move(t.shape);
		strides = std::move(t.strides);
		device = std::move(t.device);
		m_backend = std::move(t.m_backend);

		offset = t.offset; t.offset = 0;
		flags = t.flags; t.flags = 0;
		dtype = t.dtype; t.dtype = NONE;
		return *this;
	}

	explicit Tensor(const Shape& shape, const float data[], unsigned long long sz, const std::string& device = EmptyStr);
	explicit Tensor(const Shape& shape, const double data[], unsigned long long sz, const std::string& device = EmptyStr);

	explicit Tensor(const Shape& shape, const int8_t data[], unsigned long long sz, const std::string& device = EmptyStr);
	explicit Tensor(const Shape& shape, const int16_t data[], unsigned long long sz, const std::string& device = EmptyStr);
	explicit Tensor(const Shape& shape, const int32_t data[], unsigned long long sz, const std::string& device = EmptyStr);
	explicit Tensor(const Shape& shape, const int64_t data[], unsigned long long sz, const std::string& device = EmptyStr);

	explicit Tensor(const Shape& shape, const uint8_t data[], unsigned long long sz, const std::string& device = EmptyStr);
	explicit Tensor(const Shape& shape, const uint16_t data[], unsigned long long sz, const std::string& device = EmptyStr);
	explicit Tensor(const Shape& shape, const uint32_t data[], unsigned long long sz, const std::string& device = EmptyStr);
	explicit Tensor(const Shape& shape, const uint64_t data[], unsigned long long sz, const std::string& device = EmptyStr);

#ifndef SWIG
	explicit Tensor(const Shape& shape, std::initializer_list<float> data, const std::string& device = EmptyStr);
	explicit Tensor(const Shape& shape, std::initializer_list<double> data, const std::string& device = EmptyStr);

	explicit Tensor(const Shape& shape, std::initializer_list<int8_t> data, const std::string& device = EmptyStr);
	explicit Tensor(const Shape& shape, std::initializer_list<int16_t> data, const std::string& device = EmptyStr);
	explicit Tensor(const Shape& shape, std::initializer_list<int32_t> data, const std::string& device = EmptyStr);
	explicit Tensor(const Shape& shape, std::initializer_list<int64_t> data, const std::string& device = EmptyStr);

	explicit Tensor(const Shape& shape, std::initializer_list<uint8_t> data, const std::string& device = EmptyStr);
	explicit Tensor(const Shape& shape, std::initializer_list<uint16_t> data, const std::string& device = EmptyStr);
	explicit Tensor(const Shape& shape, std::initializer_list<uint32_t> data, const std::string& device = EmptyStr);
	explicit Tensor(const Shape& shape, std::initializer_list<uint64_t> data, const std::string& device = EmptyStr);

	template<typename V>
	explicit Tensor(const Shape& shape, std::initializer_list<V> data, DTYPE dtype, const std::string& device = EmptyStr):
		Tensor(shape, data, device)
	{
		*this = astype(dtype);
	}

#endif

	BackendTensor* getBackend() { return m_backend.get(); }
	
#ifndef SWIG
	inline BackendTensor* getBackend() const { return m_backend.get(); }

	inline BackendTensor* operator->() { return m_backend.get(); }
	inline BackendTensor* operator->() const { return m_backend.get(); }

	template<typename T>
	inline T* getBackendAs() { return dynamic_cast<T*>(m_backend.get()); }

	template<typename T>
	inline T* getBackendAs() const { return dynamic_cast<T*>(m_backend.get()); }

	/**
	 * @brief Casts the backend to the desired type if supported. Throws an error otherwise.
	 * @see funzel::cv::CVBackendTensor funzel::linalg::LinalgBackendTensor funzel::nn::NNBackendTensor
	 * @tparam T The desired backend type.
	 * @return T& The backend as the desired type.
	 */
	template<typename T>
	inline T& ensureBackend() const;
#endif

	std::string toString() const;

	/**
	 * @brief Checks, if a backend is currently set.
	 * @return true 
	 * @return false 
	 */
	bool empty() const { return m_backend == nullptr; }

	/**
	 * @brief Determines the number of elements of type dtype contained in the Tensor.
	 * 
	 * This basically multiplies all entries of shape together.
	 * 
	 * @return size_t The number of elements contained in the Tensor.
	 */
	size_t size() const;

	FUNZEL_INLINE size_t ndim() const { return shape.size(); }

	/**
	 * @brief Fetches the data from the given index.
	 * 
	 * The index is an array for up to shape.size() dimensions.
	 * 
	 * @param idx The index to fetch from.
	 * @return Tensor A reference to the data at the given index.
	 */
	Tensor get(const Index& idx) const;

	/**
	 * @brief Fetches the data from the given index.
	 * 
	 * @param idx The index to fetch from.
	 * @return Tensor A reference to the data at the given index.
	 */
	Tensor get(int64_t idx) const;

	/**
	 * @brief Retrieves a pointer to the data at the given offset into the Tensor.
	 * 
	 * @param offset An offset in bytes.
	 * @return void* A pointer to the data.
	 */
	void* data(size_t offset = 0);

#ifndef SWIG
	const void* data(size_t offset = 0) const;
#endif

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

	/**
	 * @brief Converts the Tensor data to another DTYPE.
	 * 
	 * @param type The new DTYPE.
	 * @return Tensor A new Tensor containing the data with the new DTYPE.
	 */
	Tensor astype(DTYPE type) const;

	/**
	 * @brief Converts the Tensor data to another DTYPE.
	 * 
	 * @tparam T The type to convert to. Needs to be supported by funzel::dtype
	 * @return Tensor A new Tensor containing the data with the new DTYPE.
	 */
	template<typename T>
	Tensor astype() const
	{
		return astype(funzel::dtype<T>());
	}

	/**
	 * @see get
	 */
	Tensor operator[](size_t idx) const { return get(idx); }

	/**
	 * @see get
	 */
	Tensor operator[](const Index& idx) const { return get(idx); }

	/**
	 * @brief Slices the Tensor.
	 * @param slices An array of TensorSlice objects defining the slices.
	 */
 	Tensor operator()(const small_vector<TensorSlice>& slices) const { return slice(slices); }
 	Tensor slice(const small_vector<TensorSlice>& slices) const;
	
	/**
	 * @brief Sets the content of the tensor to the given value.
	 * 
	 * This performs implicit type conversion from T to dtype.
	 * Only works for tensors which contain one element.
	 * 
	 * @tparam T The type of the value to set.
	 * @param t The value to set.
	 */
	template<typename T>
	void set(T t)
	{
		switch(dtype)
		{
			case DFLOAT32: ritem<float>() = t; break;
			case DUINT32: ritem<uint32_t>() = t; break;
			case DINT32: ritem<int32_t>() = t; break;
			case DFLOAT64: ritem<double>() = t; break;
			case DUINT64: ritem<uint64_t>() = t; break;
			case DINT64: ritem<int64_t>() = t; break;
			case DINT8: ritem<char>() = t; break;
			case DUINT8: ritem<unsigned char>() = t; break;
			case DINT16: ritem<int16_t>() = t; break;
			case DUINT16: ritem<uint16_t>() = t; break;

			default:
			case NONE: break;
		}
	}

	/**
	 * @brief Copies given tensor.
	 * 
	 * Shapes of the tensors must match.
	 * 
	 * @param t The tensor to copy.
	 */
	void set(const Tensor& t);
	
	/**
	 * @brief Copies the given tensor.
	 * 
	 * @see set(const Tensor& t)
	 * @see set(T t)
	 * 
	 * @tparam T The type of the value to copy.
	 * @param v The value to copy.
	 * @return Tensor& A reference to this.
	 */
	template<typename T>
	Tensor& operator=(T v) { set(v); return *this; }

	/**
	 * @brief Fetches the content of the tensor.
	 * 
	 * This performs implicit type conversion from dtype to T.
	 * Only works for tensors which contain one element.
	 * 
	 * @tparam T The desired type of the value to fetch.
	 */
	template<typename T>
	T item() const
	{
		AssertExcept(!shape.empty() || (shape.size() == 1 && shape[0] == 1), "Can only take the item of a one-element tensor!");
		const void* data = this->data(offset);

		switch(dtype)
		{
			case DFLOAT32: return *reinterpret_cast<const float*>(data); break;
			case DUINT32: return *reinterpret_cast<const uint32_t*>(data); break;
			case DINT32: return *reinterpret_cast<const int32_t*>(data); break;
			case DFLOAT64: return *reinterpret_cast<const double*>(data); break;
			case DUINT64: return *reinterpret_cast<const uint64_t*>(data); break;
			case DINT64: return *reinterpret_cast<const int64_t*>(data); break;
			case DINT16: return *reinterpret_cast<const int16_t*>(data); break;
			case DUINT16: return *reinterpret_cast<const uint16_t*>(data); break;

			case DINT8: return *reinterpret_cast<const char*>(data); break;

			default:
			case DUINT8: return *reinterpret_cast<const unsigned char*>(data); break;
		}
	}

	/**
	 * @brief Fetches a reference to the content of the tensor.
	 * 
	 * Only works for tensors which contain one element.
	 * @attention Type T and dtype have to match or else an invalid memory access may occur!
	 * 
	 * @tparam T The desired type of the reference.
	 * @return T& The reference to the data.
	 */
	template<typename T>
	T& ritem()
	{
		AssertExcept(shape.empty() || (shape.size() == 1 && shape[0] == 1), "Can only take the item of a one-element tensor!");
		void* data = this->data(offset);
		return *reinterpret_cast<T*>(data);
	}

	/**
	 * @brief Removes unitary dimensions from the Tensor.
	 * 
	 */
	void trimDimensions();

	// Operations
	/**
	 * @brief Reshapes the tensor.
	 * 
	 * Applies the new shape to this Tensor and recalculates all strides.
	 * The number of elements implied by both shapes need to be equal.
	 * 
	 * @param shape The new shape.
	 * @return Tensor The Tensor with the new shape.
	 */
	Tensor reshape(const Shape& shape);

	/**
	 * @brief Reshapes the tensor in-place.
	 *
	 * Applies the new shape to this Tensor and recalculates all strides.
	 * The number of elements implied by both shapes need to be equal.
	 * 
	 * @param shape The new shape.
	 */
	void reshape_(const Shape& shape);

	Tensor flatten()
	{
		const size_t total = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
		return reshape(Shape{total});
	}

	/**
	 * @brief Permutes the dimensions of the Tensor.
	 * 
	 * The number of dimensions in the given array and the Tensor shape
	 * need to be equal.
	 * 
	 * For example, swapping the first and second dimension of a Tensor may
	 * use the following permuation array: (1, 0, 2)
	 * 
	 * @param shape An array containing the new dimension order.
	 * @return Tensor A Tensor with the permuted shape.
	 */
	Tensor permute(const Shape& shape) const;

	/**
	 * @brief Permutes the dimensions of the Tensor in-place.
	 * @see permute
	 * @param shape An array containing the new dimension order.
	 */
	void permute_(const Shape& shape);

	/**
	 * @brief Swaps the two given axes.
	 * 
	 * Negative indices count from back to front.
	 * 
	 * @param axis1 The first axis.
	 * @param axis2 The second axis.
	 * @return Tensor A new Tensor with the two given axes swapped.
	 */
	Tensor swapaxes(int axis1, int axis2);

	/**
	 * @brief Swaps the two given axes in-place.
	 * 
	 * Negative indices count from back to front.
	 * 
	 * @see swapaxes
	 * @param axis1 The first axis.
	 * @param axis2 The second axis.
	 * @return Tensor A new Tensor with the two given axes swapped.
	 */
	void swapaxes_(int axis1, int axis2);

	/**
	 * @brief Retrieves the Tensor such that it is available on the host CPU.
	 * 
	 * If an accelerator with inaccessible memory is used, like a GPU, memory
	 * needs to be copied back to the host before it can be accessed directly.
	 * For accessible memory, no copy takes place.
	 * 
	 * @see to
	 * @return Tensor The host accessible Tensor.
	 */
	Tensor cpu() const;

	/**
	 * @brief Moves the Tensor to a specific device.
	 *
	 * If an accelerator with inaccessible memory is used, like a GPU, memory
	 * needs to be copied to the device before it can be used there.
	 * If the device is equal to the current device, no copy will be performed.
	 * 
	 * @param device The device to move the data to.
	 * @return Tensor A Tensor on the given device.
	 */
	Tensor to(const std::string& device) const;

	/**
	 * @brief Creates a new Tensor with the same shape, dtype, device and data.
	 * 
	 * @return Tensor The new Tensor.
	 */
	Tensor clone() const;

	/**
	 * @brief Re-orders the elements in memory so they are contiguous.
	 * 
	 * @return Tensor A new Tensor with re-ordered elements.
	 */
	Tensor unravel() const;

	/**
	 * @brief Transposes the Tensor.
	 * 
	 * @return Tensor A Tensor with all axes transposed.
	 */
	Tensor transpose() const;

#ifndef SWIG
	Tensor operator+(const Tensor& t) const { return add(t); }
	Tensor operator-(const Tensor& t) const { return sub(t); }
	Tensor operator*(const Tensor& t) const { return matmul(t); }
	
	Tensor operator+(double t) const { return add(Tensor::scalar_like(*this, t)); }
	Tensor operator*(double t) const { return mul(t); }
	Tensor operator/(double t) const { return mul(1.0/t); }
	Tensor operator/(const Tensor& t) const { return div(t); }

	Tensor operator-() const { return mul(-1.0); }
#endif

	/**
	 * @brief Fills the Tensor with the given value, converted to dtype.
	 * 
	 * @param value A value to fill the Tensor with.
	 * @return Tensor& A reference to this.
	 */
	Tensor& fill(double value);

	/**
	 * @brief Multiplies the Tensor with a scalar.
	 * \f$ mul(x, alpha) = x \cdot alpha \f$
	 * @return Tensor 
	 */
	Tensor mul(double alpha) const;
	Tensor& mul_(double alpha);

	Tensor mul(const Tensor& b) const;
	Tensor& mul_(const Tensor& b);

	/**
	 * @brief Calculates exponentiates the Tensor.
	 * \f$ pow(x, y) = x^y \f$
	 * @return Tensor 
	 */
	Tensor pow(const Tensor& y) const;
	Tensor& pow_(const Tensor& y);

	Tensor pow(double y) const;
	Tensor& pow_(double y);

	/**
	 * @brief Divides the Tensor element wise by another Tensor.
	 * 
	 * Performs broadcasting.
	 * 
	 * \f$ x = \{x_0, ..., x_n\},
	 * b = \{b_0, ..., b_n\},
	 * 0 \leq i \leq n: 
	 * div(x, b) = \frac{x_i}{b_i} \f$
	 * @return Tensor 
	 */
	Tensor div(const Tensor& b) const;
	Tensor& div_(const Tensor& b);

	/**
	 * @brief Performs matrix multiplication with another Tensor.
	 * 
	 * Performs broadcasting.
	 * 
	 * @return Tensor 
	 */
	Tensor matmul(const Tensor& b) const;

	/**
	 * @brief Implements a multiply add of two tensors.
	 * 
	 * Performs broadcasting.
	 * 
	 * \f$ add(x, b, alpha) = alpha \cdot x + b \f$
	 * @return Tensor 
	 */
	Tensor add(const Tensor& b, double alpha) const;
	Tensor& add_(const Tensor& b, double alpha);

	Tensor add(const Tensor& b) const;
	Tensor& add_(const Tensor& b);

	/**
	 * @brief Adds a scalar to the Tensor.
	 * \f$ add(x, alpha) = x + alpha \f$
	 * @return Tensor 
	 */
	//Tensor add(double alpha) const;
	//Tensor& add_(double alpha);

	/**
	 * @brief Subtracts a scalar from the Tensor.
	 * \f$ sub(x, alpha) = x - alpha \f$
	 * @return Tensor 
	 */
	Tensor sub(const Tensor& b, double alpha = 1.0) const;
	Tensor& sub_(const Tensor& b, double alpha = 1.0);

	/**
	 * @brief Determines the absolute values of the given Tensor.
	 * \f$ abs(x) = |x| \f$
	 * @return Tensor 
	 */
	Tensor abs() const;
	Tensor& abs_();

	/**
	 * @brief Calculates the exponential function of the Tensor.
	 * \f$ exp(x) = e^x \f$
	 * @return Tensor 
	 */
	Tensor exp() const;
	Tensor& exp_();

	/**
	 * @brief Calculates the square root of the Tensor.
	 * \f$ sqrt(x) = \sqrt{x} \f$
	 * @return Tensor 
	 */
	Tensor sqrt() const;
	Tensor& sqrt_();

	/**
	 * @brief Calculates the sine of the Tensor.
	 * @return Tensor 
	 */
	Tensor sin() const;
	Tensor& sin_();

	/**
	 * @brief Calculates the cosine of the Tensor.
	 * @return Tensor 
	 */
	Tensor cos() const;
	Tensor& cos_();

	/**
	 * @brief Calculates the tan of the Tensor.
	 * @return Tensor 
	 */
	Tensor tan() const;
	Tensor& tan_();

	/**
	 * @brief Calculates the hyperbolic tan of the Tensor.
	 * @return Tensor 
	 */
	Tensor tanh() const;
	Tensor& tanh_();

	/**
	 * @brief Calculates the sum of all elements in the Tensor.
	 * \f$ x = \{x_0, ..., x_n\}: sum(x) = \sum_{i=0}^{n} x_i \f$
	 * @return double The sum.
	 */
	//Tensor sum();

	Tensor sum(const small_vector<int>& axis = {}, DTYPE dtype = DTYPE::NONE, bool keepdims = false);

	/**
	 * @brief Checks if the Tensor is C_CONTIGUOUS.
	 * 
	 * @return true 
	 * @return false 
	 */
	bool isContiguous() const { return flags & C_CONTIGUOUS; }

	DTYPE dtype;
	Shape shape;
	Strides strides;
	size_t offset = 0;
	uint32_t flags = C_CONTIGUOUS;
	std::string device;

private:
	std::shared_ptr<BackendTensor> m_backend = nullptr;
};

#ifndef SWIG
inline Tensor operator+(double v, const Tensor& t) { return t.add(Tensor::scalar(v)); }
inline Tensor operator*(double v, const Tensor& t) { return t.mul(v); }
inline Tensor operator/(double v, const Tensor& t) { return Tensor::empty_like(t).fill(v).div_(t); }
#endif

enum POOLING_MODE
{
	MEAN_POOLING,
	MAX_POOLING
};

/**
 * @brief Defines the interface for a device specific tensor implementation.
 * 
 * This class provides the interface for many basic operations on a tensor,
 * including memory allocation and basic arithmetic.
 * 
 * A device specific implementation needs to inherit from BackendTensor and
 * may inherit other extensions like CVBackendTensor or NNBackendTensor as well
 * for more specialized functionality.
 * 
 */
class FUNZEL_API BackendTensor
{
public:
	virtual ~BackendTensor() {}

	/**
	 * @brief This method is called once when the backend is loaded.
	 * Each child class may define its own version of this method.
	 */
	static void initializeBackend() {}

	/**
	 * @brief Retrieves a pointer to the data at the given offset into the tensor.
	 * 
	 * @param offset An offset in bytes.
	 * @return void* A pointer to the data.
	 */
	virtual void* data(size_t offset = 0) = 0;
	virtual std::shared_ptr<char> buffer() = 0;

	/**
	 * @brief Clones the BackendTensor into a new object.
	 * 
	 * This is used to duplicate device specific memory.
	 * 
	 * @return std::shared_ptr<BackendTensor> The new BackendTensor.
	 */
	virtual std::shared_ptr<BackendTensor> clone() const = 0;

	/**
	 * @brief Provides a name for the backend class.
	 * 
	 * This will be used to refer to this backend, also when referring to devices
	 * supported by the backend.
	 * 
	 * @return const char* A name of the backend.
	 */
	virtual const char* backendName() const = 0;

	/**
	 * @brief Fills the memory with a scalar value.
	 * 
	 * @param self A Tensor defining which exact elements to use.
	 * @param scalar A scalar value which will be converted to the correct type.
	 */
	virtual void fill(const Tensor& self, double scalar) = 0;

	virtual void add(const Tensor& a, const Tensor& b, Tensor tgt) { UnsupportedOperationError; }

	/**
	 * @brief Calculates a multiply-add.
	 * 
	 * @param self A Tensor defining which exact elements to use.
	 * @param tgt The target tensor results will be stored to.
	 * @param alpha A multiplier.
	 * @see Tensor::mul(const Tensor&, double)
	 */
	virtual void mulAdd(const Tensor& self, Tensor tgt, double alpha) { UnsupportedOperationError; }
	
	/**
	 * @brief Multiplies elements with a scalar.
	 * 
	 * @param self A Tensor defining which exact elements to use.
	 * @param scalar A scalar value which will be converted to the correct type.
	 * @see Tensor::mul
	 */
	virtual void mul(Tensor self, double alpha) { UnsupportedOperationError; }

	/**
	 * @brief Implements matrix-matrix multiplication.
	 * 
	 * @param self A Tensor defining which exact elements to use.
	 * @param b Another matrix which will be multiplied from the right.
	 * @param tgt The target tensor results will be stored to.
	 */
	virtual void matmul(const Tensor& self, Tensor b, Tensor tgt) { UnsupportedOperationError; }

	/**
	 * @brief Divides the tensor element wise.
	 * 
	 * @param self A Tensor defining which exact elements to use.
	 * @param b A Tensor of the same shape as self.
	 * @param tgt The target tensor results will be stored to.
	 */
	virtual void div(const Tensor& self, const Tensor& b, Tensor tgt) { UnsupportedOperationError; }
	virtual void mul(const Tensor& self, const Tensor& b, Tensor tgt) { UnsupportedOperationError; }

	virtual void sub(const Tensor& self, const Tensor& b, double alpha = 1.0) { UnsupportedOperationError; }
	virtual void pow(const Tensor& self, const Tensor& y, Tensor tgt) { UnsupportedOperationError; }

	/**
	 * @brief Calculates the absolute value of elements.
	 * 
	 * @param self A Tensor defining which exact elements to use.
	 * @param tgt The target tensor results will be stored to.
	 */
	virtual void abs(const Tensor& self, Tensor tgt) { UnsupportedOperationError; }

	/**
	 * @brief Calculates the exponential function of elements.
	 * 
	 * @param self A Tensor defining which exact elements to use.
	 * @param tgt The target tensor results will be stored to.
	 * @see Tensor::exp
	 */
	virtual void exp(const Tensor& self, Tensor tgt) { UnsupportedOperationError; }

	/**
	 * @brief Calculates the square root of elements.
	 * 
	 * @param self A Tensor defining which exact elements to use.
	 * @param tgt The target tensor results will be stored to.
	 * @see Tensor::sqrt
	 */
	virtual void sqrt(const Tensor& self, Tensor tgt) { UnsupportedOperationError; }

	/**
	 * @brief Calculates the sine of elements.
	 * 
	 * @param self A Tensor defining which exact elements to use.
	 * @param tgt The target tensor results will be stored to.
	 * @see Tensor::sin
	 */
	virtual void sin(const Tensor& self, Tensor tgt) { UnsupportedOperationError; }

	/**
	 * @brief Calculates the cosine of elements.
	 * 
	 * @param self A Tensor defining which exact elements to use.
	 * @param tgt The target tensor results will be stored to.
	 * @see Tensor::cos
	 */
	virtual void cos(const Tensor& self, Tensor tgt) { UnsupportedOperationError; }

	/**
	 * @brief Calculates the tangens of elements.
	 * 
	 * @param self A Tensor defining which exact elements to use.
	 * @param tgt The target tensor results will be stored to.
	 * @see Tensor::tan
	 */
	virtual void tan(const Tensor& self, Tensor tgt) { UnsupportedOperationError; }

	/**
	 * @brief Calculates the hyperbolic tangens of elements.
	 * 
	 * @param self A Tensor defining which exact elements to use.
	 * @param tgt The target tensor results will be stored to.
	 * @see Tensor::tanh
	 */
	virtual void tanh(const Tensor& self, Tensor tgt) { UnsupportedOperationError; }

	virtual void sum(
		const Tensor& self,
		Tensor& tgt,
		const small_vector<int>& axis,
		DTYPE dtype,
		bool keepdims) { UnsupportedOperationError; }

	/**
	 * @brief Copies elements from another tensor.
	 * 
	 * @param self A Tensor defining which exact elements to overwrite.
	 * @param src A Tensor defining which exact elements to read.
	 */
	virtual void set(Tensor& self, const Tensor& src) { UnsupportedOperationError; }

	/**
	 * @brief Makes a tensor contiguous in memory.
	 * 
	 * @param self A Tensor defining which exact elements to read.
	 * @param tgt The target tensor results will be stored to.
	 */
	virtual void unravel(const Tensor& self, Tensor tgt) { UnsupportedOperationError; }

	/**
	 * @brief Calculates the arithmetic mean along the given axis.
	 * 
	 * If the axis is empty or the only entry is -1, the mean is calculated over the flattened array.
	 * 
	 * @param self The input Tensor.
	 * @param tgt The target tensor results will be stored to.
	 * @param axis The axis to calculate the mean along.
	 * @param dtype The type to use for the calculation.
	 * @param keepdims Keeps the number of dimensions such that the result can be broadcast over the input.
	 */
	virtual void mean(const Tensor& self, Tensor& tgt, const small_vector<int>& axis, DTYPE dtype, bool keepdims)
		{ UnsupportedOperationError; }

	/**
	 * @brief Converts the Tensor data to another DTYPE.
	 * 
	 * @param self The input Tensor.
	 * @param tgt The target tensor results will be stored to.
	 * @param targetType The new DTYPE.
	 */
	virtual void astype(const Tensor& self, Tensor tgt, DTYPE targetType) { UnsupportedOperationError; }

	DTYPE dtype;
	size_t size; ///< Size of the buffer in bytes.

private:
};

/**
 * @brief Calculates the number of elements defined by a given shape.
 * 
 * @param shape The shape to evaluate.
 * @return size_t The number of elements in a Tensor of given shape.
 */
inline size_t size(const Shape& shape)
{
	if(shape.empty())
		return 0;
	
	size_t sz = 1;
	for(const auto dim : shape)
		sz *= dim;

	return sz;
}

/**
 * @brief Generates a set of evenly distributed values between start and stop.
 * 
 * @param start A start value, the smallest value in the set.
 * @param stop An end value, the maximal possible value in the set (if endPoint is /b true).
 * @param num The number of values to generate.
 * @param endPoint /b true if the end point should be included in the set, /b false otherwise.
 * @param dtype The DTYPE of all elements.
 * @return Tensor A new Tensor containing the generated values.
 */
FUNZEL_API Tensor linspace(double start, double stop, size_t num, bool endPoint = true, DTYPE dtype = DFLOAT32);

/**
 * @brief Generates a set of evenly distributed values between start and stop.
 * 
 * This is the tensor version of this method, the only difference to the scalar version
 * is the dimensionality of the output.
 * 
 * @param start A start value, the smallest value in the set.
 * @param stop An end value, the maximal possible value in the set (if endPoint is /b true).
 * @param num The number of values to generate.
 * @param endPoint /b true if the end point should be included in the set, /b false otherwise.
 * @param dtype The DTYPE of all elements.
 * @return Tensor A new Tensor containing the generated values.
 */
FUNZEL_API Tensor linspace(const Tensor& start, const Tensor& stop, size_t num, bool endPoint = true, DTYPE dtype = DFLOAT32);

/**
 * @brief Generates am evenly distributed set of values between start and stop on a log scale.
 * 
 * The values start at \f$ base^{start} \f$ and end with \f$ base^{stop} \f$.
 * 
 * @param start A start value, the smallest value in the set.
 * @param stop An end value, the maximal possible value in the set (if endPoint is /b true).
 * @param num The number of values to generate.
 * @param endPoint /b true if the end point should be included in the set, /b false otherwise.
 * @param base The base of the log space.
 * @param dtype The DTYPE of all elements.
 * @return Tensor A new Tensor containing the generated values.
 */
FUNZEL_API Tensor logspace(const Tensor& start, const Tensor& stop, size_t num, bool endPoint = true, double base = 10.0, DTYPE dtype = DFLOAT32);

/**
 * @brief Generates a set of values between start and stop with a distance of step.
 * 
 * @param start A start value, the smallest value in the set.
 * @param stop An end value, the maximal possible value in the set.
 * @param step The step size between values.
 * @param dtype The DTYPE of all elements.
 * @return Tensor A new Tensor containing the generated values.
 */
FUNZEL_API Tensor arange(double start, double stop, double step, DTYPE dtype = DFLOAT32);

/**
 * @brief Calculates the arithmetic mean along the given axis.
 * 
 * If the axis is empty or the only entry is -1, the mean is calculated over the flattened array.
 * 
 * @param t The input Tensor.
 * @param axis The axis to calculate the mean along.
 * @param dtype (optional) The type to use for the calculation.
 * @param keepdims (optional) Keeps the number of dimensions such that the result can be broadcast over the input.
 * @return The mean Tensor.
 */
FUNZEL_API Tensor mean(const Tensor& t, const small_vector<int>& axis = {}, DTYPE dtype = DFLOAT32, bool keepdims = false);

/**
 * @brief Defines the interface of a general RNG.
 */
struct IRandomGenerator
{
	virtual ~IRandomGenerator() {}

	/**
	 * @brief Gets a new, implementation defined, random number.
	 * 
	 * @return double A random number.
	 */
	virtual double get() = 0;
};

/**
 * @brief A random number generator which can be used with STL distributions.
 * 
 * @tparam Dist A random number distribution type like std::uniform_distribution.
 */
template<typename Dist>
struct RandomGenerator : public IRandomGenerator
{
	RandomGenerator() = default;

	/**
	 * @brief Construct a new RandomGenerator object.
	 * 
	 * @param d A distribution object.
	 * @param seed A seed value.
	 */
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

/**
 * @brief Generates a set of random values to fill a Tensor with.
 * 
 * @param out The Tensor object to fill with random values.
 * @param generator A random number generator object.
 * @return Tensor& The out parameter for chaining operations.
 */
FUNZEL_API Tensor& randn(Tensor& out, IRandomGenerator& generator);

/**
 * @brief Generates a set of random values to fill a Tensor with.
 * 
 * @param out The Tensor object to fill with random values.
 * @return Tensor& The out parameter for chaining operations.
 */
FUNZEL_API Tensor& randn(Tensor& out);

#ifndef SWIG
FUNZEL_API std::ostream& operator<<(std::ostream& out, const Tensor& s);
FUNZEL_API std::ostream& operator<<(std::ostream& out, const Shape& s);

template<typename T>
inline std::ostream& operator<<(std::ostream& out, const small_vector<T>& s)
{
	out << "(";
	for(auto& k : s)
		out << k << (&k != &s.back() ? ", " : "");
	out << ")";
	return out;
}

// Out of line definition to prevent "invalid use of incomplete type" warning.
template<typename T>
inline T& Tensor::ensureBackend() const
{
	auto* t = dynamic_cast<T*>(m_backend.get());
	AssertExcept(t, "Expected a type supporting trait '" << T::BackendName() << "' but backend '" << m_backend->backendName() << "' does not.");
	return *t;
}

template<typename Fn>
FUNZEL_INLINE void ApplyStrided(const Tensor& self, Tensor tgt, Fn&& fn)
{
	static_assert(std::is_invocable_v<Fn, const Tensor&, Tensor>,
					"The given function needs the following signature: void(const Tensor&, Tensor)");
	
	if(self.shape.empty())
		return;
	
	if(self.shape.size() > 1)
	{
		for(int i = 0; i < self.shape[0]; i++)
		{
			ApplyStrided(self[i], (tgt.shape[0] > self.shape[0] ? tgt[i] : tgt), fn);
		}
		return;
	}

	fn(self, tgt);
}

/**
 * @brief Perform a reduction operation on a tensor along specified axes using a custom reduction function.
 *
 * This function reduces the input tensor `arr` along the specified `axis` or axes using the reduction function `ufunc`.
 * The result is stored in `out`. The reduction can optionally maintain the dimensions of the reduced axes if `keepdims`
 * is set to true. An optional initial value for the reduction can be specified with `initial`.
 *
 * @tparam T The type of the optional initial value.
 * @tparam Fn The type of the reduction function. This should be a callable that takes two `Tensor` arguments.
 *
 * @param [in] arr The input tensor to be reduced.
 * @param [in] axis A `small_vector` of integers specifying the axes along which the reduction should be performed. If empty,
 *                  the tensor is flattened before reduction.
 * @param [in] dtype The data type of the output tensor. If set to DTYPE::NONE, the data type of `arr` is used.
 * @param [out] out The output tensor where the result of the reduction is stored.
 * @param [in] keepdims If true, the reduced dimensions are retained as dimensions with size one in the output tensor.
 * @param [in] initial An optional initial value for the reduction. If provided, it is used as the starting value for the
 *                     reduction operation.
 * @param [in] ufunc The reduction function to apply. It should take two `Tensor` parameters: the source tensor and the
 *                   target tensor where results are accumulated.
 *
 * @return A reference to the output tensor `out` containing the result of the reduction.
 *
 * @note This function modifies the output tensor `out` in-place.
 *
 * @exception std::out_of_range Thrown if any axis in `axis` is out of the valid range of dimensions for `arr`.
 *
 * @b Example:
 * @code
 * Tensor myTensor; // Initialized with some data
 * Tensor outputTensor;
 * std::vector<int> axes{0}; // Reduce along the first axis
 * Reduce<float>(myTensor, axes, DTYPE::FLOAT, outputTensor, false, std::nullopt, myReductionFunction);
 * @endcode
 */
template<typename T = float, typename Fn>
FUNZEL_INLINE Tensor& Reduce(Tensor arr, const small_vector<int>& axis, DTYPE dtype, Tensor& out, bool keepdims, const std::optional<T>& initial, Fn&& ufunc) // TODO: Initial
{
	static_assert(std::is_invocable_r_v<void, Fn, Tensor, Tensor&>, "Functor needs to have signature compatible to void(Tensor, Tensor)!");
	
	// Step 1: Validate and prepare input arguments
	Shape origShape = arr.shape; // In case of keepdims this is required.
	if(axis.empty())
	{
		arr = arr.flatten();
	}
	else
	{
		if(axis.size() == 1) // Recursion anchor
		{
			arr = arr.swapaxes(axis.front(), -1);
		}
		else
		{
			auto sortedAx = axis;
			std::sort(sortedAx.begin(), sortedAx.end(), std::greater<int>());

			Tensor input = arr;
			for(int ax : sortedAx) // Reduce for each dimension
			{
				Reduce(input, {ax}, dtype, out, keepdims, initial, ufunc);
				input = std::move(out);
			}

			out = std::move(input);
			return out;
		}
	}

	if(dtype == DTYPE::NONE)
		dtype = arr.dtype;

	if(out.empty())
	{
		auto newShape = arr.shape;
		newShape.pop_back();

		if(newShape.empty())
			newShape.push_back(1); // Scalar tensors do not exist here as they do in Numpy!

		out = Tensor::empty(newShape, dtype, arr.device);
	}

	// Step 2: Perform the reduction operation
	//if initial is not <no value>:
	if(initial.has_value())
		out.fill(*initial);

	const auto arrT = arr.transpose();
	auto outT = out.transpose();
	for(size_t i = 0; i < arr.shape.back(); i++)
	{
		ufunc(arrT[i], outT);
	}

	// TODO Find out why this is required only for axis=0
	if(axis.front() == 0)
		out = outT;

	// Step 3: Handle keepdims option
	if(keepdims)
	{
		if(axis.empty())
		{
			// Need to use original shape because arr has been flattened.
			Shape shape(origShape.size(), 1);
			std::fill(shape.begin(), shape.end(), 1);
			out = out.reshape(shape);
		}
		else
		{
			Shape shape(arr.shape);
			shape.back() = 1;
			out = out.reshape(shape);
		}
	}

	return out;
}

template<typename Fn>
FUNZEL_INLINE void ApplyStridedAsType(const Tensor& self, Tensor tgt, DTYPE dtype, Fn&& fn)
{
	if(self.shape.empty())
		return;
	
	if(self.shape.size() > 1)
	{
		for(int i = 0; i < self.shape[0]; i++)
		{
			//std::cout << tgt.shape << "   " << self.shape << std::endl;
			ApplyStridedAsType(self[i], (tgt.shape[0] >= self.shape[0] ? tgt[i] : tgt), dtype, fn);
		}
		return;
	}

	DoAsDtype(dtype, fn, self, tgt);
}

template<typename Fn>
FUNZEL_INLINE void ApplyStrided(const Tensor& a, const Tensor& b, Tensor tgt, Fn&& fn)
{
	static_assert(std::is_invocable_v<Fn, const Tensor&, const Tensor&, Tensor>,
					"The given function needs the following signature: void(const Tensor&, Tensor)");
	
	if(a.shape.empty())
		return;
	
	if(a.shape.size() > 1)
	{
		for(int i = 0; i < a.shape[0]; i++)
		{
			ApplyStrided(a[i], b[i], tgt[i], fn);
		}
		return;
	}

	fn(a, b, tgt);
}

inline bool IsBroadcastable(const Shape& a, const Shape& b)
{
	if(b.size() >= a.size())
		return false;

	for(int i = 1; i <= b.size(); i++)
	{
		const auto bidx = b.size() - i;
		const auto aidx = a.size() - i;
		if(b[bidx] != a[aidx] && b[bidx] != 1 && a[aidx] != 1)
			return false;
	}

	return true;
}

template<typename Fn, typename... Args>
inline void Apply(const Tensor& a, const Tensor& b, Tensor tgt, size_t stopDim, Fn fn, Args&&... args)
{
	static_assert(std::is_invocable_v<Fn, const Tensor&, const Tensor&, Tensor&, Args&&...>,
					"The given function needs the following signature: void(const Tensor&, const Tensor&, Tensor&, Args&&...)");

	if(a.shape.size() < stopDim)
		return;

	if(a.shape.size() == stopDim)
	{
		fn(a, b, tgt, std::forward<Args>(args)...);
		return;
	}

	for(int64_t i = 0; i < a.shape[0]; i++)
	{
		Apply(a[i], b, tgt[i], stopDim, fn, std::forward<Args>(args)...);
	}
}

template<unsigned int StopDims = 1, typename Fn, typename... Args>
inline void Broadcast(const Tensor& a, const Tensor& b, Tensor& tgt,
	std::function<Shape(const Shape&, const Shape&)> determineShape,
	Fn fn, Args&&... args)
{
	if(a.shape.empty() || b.shape.empty())
		return;

	// Check if tensors are trivially fitting
	if(a.shape.size() == b.shape.size())
	{
		if(tgt.empty())
		{
			Shape ashape = a.shape;
			ashape.erase(ashape.begin(), ashape.begin() + a.shape.size() - b.shape.size());

			const Shape newShape = determineShape(ashape, b.shape);
			tgt = Tensor::zeros(newShape, a.dtype, a.device);
			tgt.trimDimensions();
		}

		fn(a, b, tgt, std::forward<Args>(args)...);
		return;
	}

	AssertExcept(IsBroadcastable(a.shape, b.shape), "Cannot broadcast " << b.shape << " to " << a.shape);
	if(tgt.empty())
	{
		Shape ashape = a.shape;
		ashape.erase(ashape.begin(), ashape.begin() + a.shape.size() - b.shape.size());

		const Shape newShape = determineShape(ashape, b.shape);
		
		ashape = a.shape;
		ashape.erase(ashape.begin() + a.shape.size() - b.shape.size(), ashape.end());

		for(auto d : newShape)
			ashape.push_back(d);

		tgt = Tensor::zeros(ashape, a.dtype, a.device);
		tgt.trimDimensions();
	}

	Apply(a, b, tgt, tgt.shape.size() - b.shape.size() + StopDims, fn, std::forward<Args>(args)...);
}

#endif

}
