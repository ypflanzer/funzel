#include "funzel/Type.hpp"
#include <funzel/Tensor.hpp>
#include <funzel/Funzel.hpp>
#include <functional>

#include <spdlog/spdlog.h>
#include <sstream>
#include <algorithm>

using namespace funzel;

void outputTensor(std::ostream& out, const Tensor& s, const std::string& prefix = "")
{
	if(s.shape.empty())
	{
		out << "[]";
		return;
	}

	if(s.shape.size() == 1)
	{
		if(s.shape[0] > 1)
			out << "[";

		size_t count = std::min(s.shape[0], size_t(10));
		for(size_t i = 0; i < count; i++)
		{
			out << s[i].item<double>() << (i < count - 1 ? ", " : "");
		}

		if(s.shape[0] >= 10)
			out << ", ...";

		if(s.shape[0] > 1)
			out << "]";

		return;
	}

	out << "[";
	for(size_t i = 0; i < std::min(s.shape[0], size_t(10)); i++)
	{
		if(i > 0)
		{
			if(s.shape[1] > 1)
				out << ",\n" << prefix;
			else
				out << ", ";
		}
		
		outputTensor(out, s[i], prefix);
	}

	if(s.shape[0] >= 10)
	{
		if(s.shape[1] != 1)
			out << ",\n" << prefix << "[...]";
		else
			out << ", ...";
	}
	
	out << "]";
}

std::ostream& funzel::operator<<(std::ostream& out, const Tensor& s)
{
	if(s.shape.empty())
		return out;

	out << "Tensor<" << dtypeToNativeString(s.dtype) << ", " << s.shape << ">(";
	outputTensor(out, s, "\t");
	out << ")";
	return out;
}

std::ostream& funzel::operator<<(std::ostream& out, const Shape& s)
{
	out << "(";
	for(auto& k : s)
		out << k << (&k != &s.back() ? ", " : "");
	out << ")";
	return out;
}

Tensor::Tensor(const Shape& shape, const float data[], unsigned long long sz, const std::string& device)
{
	AssertExcept(::size(shape) == sz, "Given shape does not match the size of the initial data!");
	*this = empty(shape, (const void*) data, DFLOAT32, device);
}

Tensor::Tensor(const Shape& shape, const double data[], unsigned long long sz, const std::string& device)
{
	AssertExcept(::size(shape) == sz, "Given shape does not match the size of the initial data!");
	*this = empty(shape, (const void*) data, DFLOAT64, device);
}

Tensor::Tensor(const Shape& shape, const int8_t data[], unsigned long long sz, const std::string& device)
{
	AssertExcept(::size(shape) == sz, "Given shape does not match the size of the initial data!");
	*this = empty(shape, (const void*) data, DINT8, device);
}

Tensor::Tensor(const Shape& shape, const int16_t data[], unsigned long long sz, const std::string& device)
{
	AssertExcept(::size(shape) == sz, "Given shape does not match the size of the initial data!");
	*this = empty(shape, (const void*) data, DINT16, device);
}

Tensor::Tensor(const Shape& shape, const int32_t data[], unsigned long long sz, const std::string& device)
{
	AssertExcept(::size(shape) == sz, "Given shape does not match the size of the initial data!");
	*this = empty(shape, (const void*) data, DINT32, device);
}

Tensor::Tensor(const Shape& shape, const int64_t data[], unsigned long long sz, const std::string& device)
{
	AssertExcept(::size(shape) == sz, "Given shape does not match the size of the initial data!");
	*this = empty(shape, (const void*) data, DINT64, device);
}

Tensor::Tensor(const Shape& shape, const uint8_t data[], unsigned long long sz, const std::string& device)
{
	AssertExcept(::size(shape) == sz, "Given shape does not match the size of the initial data!");
	*this = empty(shape, (const void*) data, DUINT8, device);
}

Tensor::Tensor(const Shape& shape, const uint16_t data[], unsigned long long sz, const std::string& device)
{
	AssertExcept(::size(shape) == sz, "Given shape does not match the size of the initial data!");
	*this = empty(shape, (const void*) data, DUINT16, device);
}

Tensor::Tensor(const Shape& shape, const uint32_t data[], unsigned long long sz, const std::string& device)
{
	AssertExcept(::size(shape) == sz, "Given shape does not match the size of the initial data!");
	*this = empty(shape, (const void*) data, DUINT32, device);
}

Tensor::Tensor(const Shape& shape, const uint64_t data[], unsigned long long sz, const std::string& device)
{
	AssertExcept(::size(shape) == sz, "Given shape does not match the size of the initial data!");
	*this = empty(shape, (const void*) data, DUINT64, device);
}

Tensor::Tensor(const Shape& shape, const std::initializer_list<double> data, const std::string& device):
	Tensor(shape, data.begin(), data.size(), device) {}

Tensor::Tensor(const Shape& shape, const std::initializer_list<float> data, const std::string& device):
	Tensor(shape, data.begin(), data.size(), device) {}

Tensor::Tensor(const Shape& shape, const std::initializer_list<int8_t> data, const std::string& device):
	Tensor(shape, data.begin(), data.size(), device) {}

Tensor::Tensor(const Shape& shape, const std::initializer_list<int16_t> data, const std::string& device):
	Tensor(shape, data.begin(), data.size(), device) {}

Tensor::Tensor(const Shape& shape, const std::initializer_list<int32_t> data, const std::string& device):
	Tensor(shape, data.begin(), data.size(), device) {}

Tensor::Tensor(const Shape& shape, const std::initializer_list<int64_t> data, const std::string& device):
	Tensor(shape, data.begin(), data.size(), device) {}

Tensor::Tensor(const Shape& shape, const std::initializer_list<uint8_t> data, const std::string& device):
	Tensor(shape, data.begin(), data.size(), device) {}

Tensor::Tensor(const Shape& shape, const std::initializer_list<uint16_t> data, const std::string& device):
	Tensor(shape, data.begin(), data.size(), device) {}

Tensor::Tensor(const Shape& shape, const std::initializer_list<uint32_t> data, const std::string& device):
	Tensor(shape, data.begin(), data.size(), device) {}

Tensor::Tensor(const Shape& shape, const std::initializer_list<uint64_t> data, const std::string& device):
	Tensor(shape, data.begin(), data.size(), device) {}

std::string Tensor::toString() const
{
	std::stringstream ss;
	ss << *this;
	return ss.str();
}

Tensor Tensor::ones(size_t count, DTYPE dtype, const std::string& backend)
{
	return Tensor::ones(Shape{count}, dtype, backend);
}

Tensor Tensor::zeros(size_t count, DTYPE dtype, const std::string& backend)
{
	return Tensor::zeros(Shape{count}, dtype, backend);
}

Tensor Tensor::empty(const Shape& shape, const std::shared_ptr<char>& data, DTYPE dtype, const std::string& backend)
{
	Tensor t;
	const auto count = ::size(shape);
	t.m_backend = backend::CreateBackendTensor(data, count, dtype, backend);

	if(!t.m_backend)
		throw std::runtime_error("Could not create tensor with device '" + backend + "'!");
		
	t.dtype = dtype;
	t.shape = shape;
	t.device = backend;
	
	// Reshape to set the strides
	t.reshape_(shape);

	return t;
}

Tensor Tensor::empty(const Shape& shape, const void* data, DTYPE dtype, const std::string& backend)
{
	Tensor t;
	const auto count = ::size(shape);
	t.m_backend = backend::CreateBackendTensor(data, count, dtype, backend);

	if(!t.m_backend)
		throw std::runtime_error("Could not create tensor with device '" + backend + "'!");
		
	t.dtype = dtype;
	t.shape = shape;
	t.device = backend;

	// Reshape to set the strides
	t.reshape_(shape);
	return t;
}

Tensor Tensor::empty(const Shape& shape, DTYPE dtype, const std::string& backend)
{
	return empty(shape, nullptr, dtype, backend);
}

Tensor Tensor::empty_like(const Tensor& t)
{
	auto nt = empty(t.shape, t.dtype, t.device);
	// nt.strides = t.strides;
	return nt;
}

Tensor Tensor::ones(const Shape& shape, DTYPE dtype, const std::string& backend)
{
	Tensor t = Tensor::empty(shape, dtype, backend);
	t->fill(t, 1.0);
	return t;
}

Tensor Tensor::ones_like(const Tensor& t)
{
	auto nt = ones(t.shape, t.dtype, t.device);
	// nt.strides = t.strides;
	return nt;
}

Tensor Tensor::zeros(const Shape& shape, DTYPE dtype, const std::string& backend)
{
	Tensor t = Tensor::empty(shape, dtype, backend);
	t->fill(t, 0.0);
	return t;
}

Tensor Tensor::zeros_like(const Tensor& t)
{
	auto nt = zeros(t.shape, t.dtype, t.device);
	//nt.strides = t.strides;
	return nt;
}

size_t Tensor::size() const
{
	return ::size(shape);
}

Tensor Tensor::get(int64_t idx) const
{
	AssertExcept(!shape.empty(), "Cannot index an empty tensor!");
	AssertExcept(std::abs(idx) < shape[0], "Index out of bounds error: " << idx << " >= " << shape[0]);

	Tensor t(*this);
	t.shape.erase(t.shape.begin());
	t.strides.erase(t.strides.begin());

	// In case a specific element is requested, ensure the shape is not empty!
	if(t.shape.empty())
	{
		t.strides.push_back(dtypeSizeof(dtype));
		t.shape.push_back(1);
	}

	// Handle negative steps in strides
	int64_t newOffset = t.offset + idx * strides[0];
	if(newOffset < 0)
		newOffset = t->size + newOffset;

	t.offset = newOffset;
	assert(t.offset < t->size);
	return t;
}

Tensor Tensor::get(const Index& idx) const
{
	Tensor t = *this;
	for(size_t i : idx)
	{
		t = t[i];
	}
	return t;
}

Tensor Tensor::slice(const small_vector<TensorSlice>& slices) const
{
	Tensor t = *this;

	AssertExcept(slices.size() <= shape.size(), "Too many indices for tensor of dimension " << shape.size());
	for(size_t i = 0; i < slices.size(); i++)
	{
		const auto& s = slices[i];
		int64_t first = s.first;
		int64_t last = s.last;
		int64_t step = s.step;

		// Handle negative indices
		if(first < 0) first = t.shape[i] + first;
		if(last < 0) last = t.shape[i] + last;

		// Check out of bounds for the start element
		if(first < 0 || first >= t.shape[i])
			throw std::out_of_range("Slice start index out of range: " + std::to_string(s.first) + " for dimension " + std::to_string(i) + " with size " + std::to_string(t.shape[i]));
		
		// Check out of bounds for the end element
		if(last < 0 || last > t.shape[i])
			throw std::out_of_range("Slice end index out of range: " + std::to_string(s.last) + " for dimension " + std::to_string(i) + " with size " + std::to_string(t.shape[i]));

		if(step == 0)
			throw std::invalid_argument("Slice step cannot be zero!");

		if(first >= last)
			throw std::invalid_argument("Slice start index must be less than end index!");

		// FIXME Strange that it only works with s.last == -1, probably because '[...]/step' is rounded toward zero
		//       by default.
		const int64_t newSize = (last - first + std::abs(step) - 1) / std::abs(step) + (s.last == -1 ? 1 : 0);
		const int64_t oldStride = t.strides[i];

		// Set the new values
		t.shape[i] = newSize;
		t.strides[i] = oldStride * step;
		t.offset += first * oldStride;
	}

	return t;
}

void* Tensor::data(size_t offset)
{
	return m_backend->data(offset);
}

const void* Tensor::data(size_t offset) const
{
	return m_backend->data(offset);
}

void Tensor::trimDimensions()
{
	// TODO The elements should be at the same positions, use that!
	//shape.erase(std::unique(shape.begin(), shape.end()));
	//strides.erase(std::unique(strides.begin(), strides.end()));
	for(size_t i = 0; i < shape.size() - 1; i++)
	{
		if(shape[i] == 1 && shape[i] == shape[i + 1])
		{
			shape.erase(shape.begin() + i + 1);
			strides.erase(strides.begin() + i + 1);
		}
	}

}

template<typename From, typename To>
static void convertType(const Tensor& in, Tensor& tgt)
{
	for(int64_t i = 0; i < tgt.size(); i++)
	{
		tgt.dataAs<To>(i) = in.dataAs<From>(i);
	}
}

template<typename From>
static void convertType(DTYPE to, const Tensor& in, Tensor& tgt)
{
	switch(to)
	{
		case DFLOAT32: convertType<From, float>(in, tgt); break;
		case DUINT32: convertType<From, uint32_t>(in, tgt); break;
		case DINT32: convertType<From, int32_t>(in, tgt); break;
		case DFLOAT64: convertType<From, double>(in, tgt); break;
		case DUINT64: convertType<From, uint64_t>(in, tgt); break;
		case DINT64: convertType<From, int64_t>(in, tgt); break;
		case DINT8: convertType<From, char>(in, tgt); break;
		case DUINT8: convertType<From, unsigned char>(in, tgt); break;
		case DINT16: convertType<From, int16_t>(in, tgt); break;
		case DUINT16: convertType<From, uint16_t>(in, tgt); break;
		case NONE: break;
		default: throw std::invalid_argument("Unsupported DTYPE given: " + std::to_string(to));
	}
}

Tensor Tensor::astype(DTYPE type) const
{
	Tensor t = empty(this->shape, type, this->device);
	m_backend->astype(*this, t, type);
	return t;

}

Tensor Tensor::to(const std::string& device) const
{
	// If the tensor is completely empty, return an empty tensor.
	if(!m_backend)
		return {};

	// If we do not change device, return self
	if((device.empty() && m_backend->backendName() == funzel::GetDefaultBackend())
		|| m_backend->backendName() == device)
	{
		return *this;
	}

	Tensor t;
	const auto count = ::size(shape);

	// TODO Optimize if the backend is the same, e.g. OpenCL
	t.m_backend = backend::CreateBackendTensor(m_backend->buffer(), count, dtype, device);

	if(!t.m_backend)
		throw std::runtime_error("Could not create tensor with device '" + device + "'!");

	t.dtype = dtype;
	t.shape = shape;
	t.device = device;
	t.offset = offset;
	t.strides = strides;
	
	//t->empty(m_backend->buffer(), m_backend->size, dtype);
	// Reshape to set the strides
	// t.reshape_(shape);
	return t;
}

Tensor Tensor::cpu() const
{
	return to("");
}

void Tensor::set(const Tensor& t)
{
	AssertExcept(shape == t.shape, "Cannot copy tensor to target with different shape: " << t.shape << " vs " << shape);
	m_backend->set(*this, t.to(device));
}

Tensor Tensor::reshape(const Shape& shape)
{
	Tensor t(*this);
	t.reshape_(shape);
	return t;
}

void Tensor::reshape_(const Shape& shape)
{
	AssertExcept(::size(shape) == ::size(this->shape),
		"Cannot reshape due to a size conflict: " + std::to_string(::size(shape)) + " vs " + std::to_string(::size(this->shape)));

	this->shape = shape;
	this->strides.resize(shape.size());

	size_t offset = 1;
	for(int i = shape.size() - 1; i >= 0; i--)
	{
		strides[i] = offset*dtypeSizeof(dtype);
		offset *= shape[i];
	}
}

Tensor Tensor::permute(const Shape& shape) const
{
	Tensor t(*this);
	t.permute_(shape);
	return t;
}

void Tensor::permute_(const Shape& indices)
{
	AssertExcept(indices.size() == this->shape.size(),
		"Cannot permute axes, given order has a different number of dimensions: " << indices.size() << " vs " << this->shape.size());

	const auto oldstrides = strides;
	const auto oldshape = shape;

	for(size_t i = 0; i < indices.size(); i++)
	{
		if(indices[i] >= shape.size())
			throw std::out_of_range("Cannot permute axes, given index value exceeds the number of dimensions!");

		shape[i] = oldshape[indices[i]];
		strides[i] = oldstrides[indices[i]];
	}

	flags &= (~C_CONTIGUOUS);
}

Tensor Tensor::swapaxes(int axis1, int axis2)
{
	Shape nshape(shape.size());

	if(axis1 < 0) axis1 = shape.size() - axis1 - 2;
	if(axis2 < 0) axis2 = shape.size() - axis2 - 2;

	std::iota(nshape.begin(), nshape.end(), 0);
	std::swap(nshape[axis1], nshape[axis2]);

	return permute(nshape);
}

void Tensor::swapaxes_(int axis1, int axis2)
{
	Shape nshape(shape.size());

	if(axis1 < 0) axis1 = shape.size() - axis1;
	if(axis2 < 0) axis2 = shape.size() - axis2;

	std::iota(nshape.begin(), nshape.end(), 0);
	std::swap(nshape[axis1], nshape[axis2]);
	
	permute_(nshape);
}

Tensor Tensor::unravel() const
{
	Tensor t = Tensor::empty(shape, dtype, device);
	m_backend->unravel(*this, t);
	return t;
}

Tensor Tensor::clone() const
{
	Tensor t;
	t.m_backend = m_backend->clone();
	t.dtype = dtype;
	t.shape = shape;
	t.flags = flags;
	t.device = device;

	// Reshape to set the strides
	t.reshape_(shape);
	return t;
}

Tensor Tensor::transpose() const
{
	Tensor t(*this);
	std::reverse(t.strides.begin(), t.strides.end());
	std::reverse(t.shape.begin(), t.shape.end());

	// Unset contiguous flag!
	t.flags &= ~C_CONTIGUOUS;
	return t;
}

Tensor& Tensor::fill(double value)
{
	m_backend->fill(*this, value);
	return *this;
}

Tensor& Tensor::add_(const Tensor& b)
{
	m_backend->add(*this, b, *this);
	return *this;
}

Tensor Tensor::add(const Tensor& b) const
{
	Tensor t = Tensor::empty_like(*this);
	t.m_backend->add(*this, b, t);
	return t;
}

Tensor& Tensor::add_(const Tensor& b, double alpha)
{
	Broadcast<0>(*this, b, *this,
		[](const auto& a, const auto& b) { return b; },
		[](const Tensor& a, Tensor b, Tensor c, double alpha) {
			b->mulAdd(b, c, alpha);
		}, alpha);

	return *this;
}

Tensor Tensor::add(const Tensor& b, double alpha) const
{
	// TODO Run without copy!!!
	Tensor t = clone();
	t.add_(b, alpha);
	return t;
}

Tensor& Tensor::sub_(const Tensor& b, double alpha)
{
	return add_(b, -alpha);
}

Tensor Tensor::sub(const Tensor& b, double alpha) const
{
	// TODO Run without copy!!!
	Tensor t = clone();
	t.add_(b, -alpha);
	return t;
}

Tensor& Tensor::mul_(double alpha)
{
	m_backend->mul(*this, alpha);
	return *this;
}

Tensor Tensor::mul(double alpha) const
{
	// TODO Run without copy!!!
	Tensor t = clone();
	t.mul_(alpha);
	return t;
}

Tensor Tensor::mul(const Tensor& b) const
{
	// TODO Run without copy!!!
	Tensor t = clone();
	t.mul_(b);
	return t;
}

Tensor& Tensor::mul_(const Tensor& b)
{
	Broadcast<0>(*this, b, *this,
		[](const auto& a, const auto& b) { return b; },
		[](Tensor a, Tensor b, Tensor c) {
			a->mul(a, b, c);
		});

	return *this;
}

Tensor Tensor::div(const Tensor& b) const
{
	// TODO Run without copy!!!
	Tensor t = clone();
	t.div_(b);
	return t;
}

Tensor& Tensor::div_(const Tensor& b)
{
	Broadcast<0>(*this, b, *this,
		[](const auto& a, const auto& b) { return b; },
		[](Tensor a, Tensor b, Tensor c) {
			a->div(a, b, c);
		});

	return *this;
}

Tensor& Tensor::pow_(double y)
{
	m_backend->mul(*this, y);
	return *this;
}

Tensor Tensor::pow(double y) const
{
	// TODO Run without copy!!!
	Tensor t = clone();
	t.pow_(y);
	return t;
}

Tensor& Tensor::pow_(const Tensor& y)
{
	Broadcast<0>(*this, y, *this,
		[](const auto& a, const auto& b) { return b; },
		[](Tensor a, Tensor b, Tensor c) {
			a->pow(a, b, c);
		});

	return *this;
}

Tensor Tensor::pow(const Tensor& y) const
{
	Tensor t = clone();
	t.pow_(y);
	return t;
}

Tensor Tensor::matmul(const Tensor& b) const
{
	AssertExcept(dtype == b.dtype,
		"Cannot multiply matrices with different dtypes: " << dtypeToNativeString(dtype) << " vs " << dtypeToNativeString(b.dtype));
	
	Tensor tgt;
	Broadcast<2>(*this, b, tgt,
		[](const Shape& a, const Shape& b) {
			AssertExcept(a.size() >= 2 && a.size() == b.size(), "Invalid matrix shapes: Number of dimensions do not match!");

			auto* matShapeA = &a[a.size() - 2];
			auto* matShapeB = &b[b.size() - 2];
			AssertExcept(matShapeB[0] == matShapeA[1], "Invalid matrix shapes: " << matShapeB[0] << " != " << matShapeA[1]);

			Shape newshape(a);
			newshape[newshape.size() - 1] = matShapeB[1];
			newshape[newshape.size() - 2] = matShapeA[0];
			return newshape;
		},
		[](Tensor a, Tensor b, Tensor c) {
			a->matmul(a, b, c);
		});

	return tgt;
}

Tensor Tensor::sum(const small_vector<int>& axis, DTYPE dtype, bool keepdims)
{
	if(dtype == NONE)
		dtype = this->dtype;

	Tensor t;// = Tensor::empty({1}, dtype, device);
	m_backend->sum(*this, t, axis, dtype, keepdims);
	return t;
}

#define UNARY_OP_PAIR(f) \
Tensor& Tensor::f##_() \
{ \
	m_backend->f(*this, *this); \
	return *this; \
} \
Tensor Tensor::f() const \
{ \
	Tensor t = clone(); \
	t.f##_(); \
	return t; \
}

UNARY_OP_PAIR(abs)
UNARY_OP_PAIR(exp)
UNARY_OP_PAIR(sqrt)
UNARY_OP_PAIR(sin)
UNARY_OP_PAIR(cos)
UNARY_OP_PAIR(tan)
UNARY_OP_PAIR(tanh)

Tensor funzel::linspace(double start, double stop, size_t num, bool endPoint, DTYPE dtype)
{
	Tensor tstart = Tensor::empty({1}, dtype);
	Tensor tend = Tensor::empty({1}, dtype);

	tstart = start;
	tend = stop;

	return linspace(tstart, tend, num, endPoint, dtype);
}

Tensor funzel::linspace(const Tensor& start, const Tensor& stop, size_t num, bool endPoint, DTYPE dtype)
{
	AssertExcept(start.shape == stop.shape,
				"Cannot calculate range for tensors that do not match shape: " << start.shape << " vs " << stop.shape);
	
	AssertExcept(num > 0,
				"Cannot calculate range for tensors with a sample size of zero!");

	if(!endPoint)
		num--;

	auto delta = (stop - start).mul_(1.0/num);

	auto shape = start.shape;
	shape.emplace(shape.begin(), num);

	// TODO Make parallel!
	auto iter = start;
	Tensor t = Tensor::empty(shape, dtype);
	for(int64_t i = 0; i < t.shape[0]; i++)
	{
		t[i].set(iter);
		iter.add_(delta);
	}

	return t;
}

Tensor funzel::logspace(const Tensor& start, const Tensor& stop, size_t num, bool endPoint, double base, DTYPE dtype)
{
	UnsupportedOperationError;
	
	Tensor t;
	return t;
}

Tensor funzel::arange(double start, double stop, double step, DTYPE dtype)
{
	AssertExcept(step > 0,
				"Cannot calculate range with a step size of zero!");

	size_t steps = (stop - start) / step;
	Tensor t = Tensor::empty({steps}, dtype);

	#pragma omp parallel for
	for(int64_t i = 0; i < t.shape[0]; i++)
	{
		t[i].set(start + step*i);
	}

	return t;
}

template<typename T>
void FillRandom(void* data, IRandomGenerator& gen, size_t count)
{
	#pragma omp parallel for
	for(int64_t i = 0; i < count; i++)
	{
		reinterpret_cast<T*>(data)[i] = gen.get();
	}
}

Tensor& funzel::randn(Tensor& out, IRandomGenerator& generator)
{
	// FIXME This should respect the shape and given ranges!
	AssertExcept(out.isContiguous(), "Random fill is only implemented for contiguous tensors!");

	const auto sz = out.size();
	switch(out.dtype)
	{
		case DINT32: FillRandom<int32_t>(out.data(out.offset), generator, sz); break;
		case DINT64: FillRandom<int64_t>(out.data(out.offset), generator, sz); break;
		case DFLOAT32: FillRandom<float>(out.data(out.offset), generator, sz); break;
		case DFLOAT64: FillRandom<double>(out.data(out.offset), generator, sz); break;
		case DUINT32: FillRandom<uint32_t>(out.data(out.offset), generator, sz); break;
		case DUINT64: FillRandom<uint64_t>(out.data(out.offset), generator, sz); break;
		case DINT8: FillRandom<char>(out.data(out.offset), generator, sz); break;
		case DUINT8: FillRandom<unsigned char>(out.data(out.offset), generator, sz); break;
		default: ThrowError("Uknown dtype!");
	}

	return out;
}

Tensor& funzel::randn(Tensor& out)
{
	RandomGenerator<std::uniform_real_distribution<double>> gen{std::uniform_real_distribution<double>(-1.0, 1.0)};
	return randn(out, gen);
}

Tensor funzel::mean(const Tensor& t, const small_vector<int>& axis, DTYPE dtype, bool keepdims)
{
	if(t.empty())
		return {};

	Tensor tgt;// = Tensor::empty({t.shape[axis[0]]}, t.dtype, t.device);
	t.getBackend()->mean(t, tgt, axis, dtype, keepdims);
	return tgt;
}
