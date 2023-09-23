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

#include "small_vector"

#include <memory>
#include <vector>
#include <random>

namespace funzel
{

typedef small_vector<size_t> Shape;
typedef small_vector<size_t> Index;
//typedef std::vector<size_t> Shape;
//typedef std::vector<size_t> Index;

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

	Tensor() = default;
	explicit Tensor(const Shape& shape, const float data[], unsigned long long sz, const std::string& device = EmptyStr);
	explicit Tensor(const Shape& shape, const double data[], unsigned long long sz, const std::string& device = EmptyStr);

#ifndef SWIG
	explicit Tensor(const Shape& shape, std::initializer_list<float> data, const std::string& device = EmptyStr);
	explicit Tensor(const Shape& shape, std::initializer_list<double> data, const std::string& device = EmptyStr);
#endif

	BackendTensor* getBackend() { return m_backend.get(); }
	
#ifndef SWIG
	inline const BackendTensor* getBackend() const { return m_backend.get(); }

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
	Tensor get(size_t idx) const;

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
		AssertExcept(shape.empty() || (shape.size() == 1 && shape[0] == 1), "Can only take the item of a one-element tensor!");
		const void* data = this->data(offset);

		switch(dtype)
		{
			case DFLOAT32: return *reinterpret_cast<const float*>(data); break;
			case DUINT32: return *reinterpret_cast<const uint32_t*>(data); break;
			case DINT32: return *reinterpret_cast<const int32_t*>(data); break;
			case DFLOAT64: return *reinterpret_cast<const double*>(data); break;
			case DUINT64: return *reinterpret_cast<const uint64_t*>(data); break;
			case DINT64: return *reinterpret_cast<const int64_t*>(data); break;
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
	
	Tensor operator+(double t) const { return add(t); }
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
	Tensor add(const Tensor& b, double alpha = 1.0) const;
	Tensor& add_(const Tensor& b, double alpha = 1.0);

	/**
	 * @brief Adds a scalar to the Tensor.
	 * \f$ add(x, alpha) = x + alpha \f$
	 * @return Tensor 
	 */
	Tensor add(double alpha) const;
	Tensor& add_(double alpha);

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
	double sum();

	/**
	 * @brief Checks if the Tensor is C_CONTIGUOUS.
	 * 
	 * @return true 
	 * @return false 
	 */
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

	virtual void sub(const Tensor& self, const Tensor& b, double alpha = 1.0) { UnsupportedOperationError; }
	
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

	/**
	 * @brief Calculates the sum of elements.
	 * 
	 * @param self A Tensor defining which exact elements to use.
	 * @return double The sum of values contained in Tensor self.
	 */
	virtual double sum(const Tensor& self) { UnsupportedOperationError; return 0; }

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

	DTYPE dtype;
	size_t size;

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


// Out of line definition to prevent "invalid use of incomplete type" warning.
template<typename T>
inline T& Tensor::ensureBackend() const
{
	auto* t = dynamic_cast<T*>(m_backend.get());
	AssertExcept(t, "Expected a type supporting trait '" << T::BackendName() << "' but backend '" << m_backend->backendName() << "' does not.");
	return *t;
}

#endif

}
