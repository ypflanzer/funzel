#include <funzel/Tensor.hpp>
#include <funzel/Funzel.hpp>
#include <functional>

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

	out << "Tensor(";
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
	*this = empty(shape, (const void*) data, FLOAT32, device);
}

Tensor::Tensor(const Shape& shape, const double data[], unsigned long long sz, const std::string& device)
{
	AssertExcept(::size(shape) == sz, "Given shape does not match the size of the initial data!");
	*this = empty(shape, (const void*) data, FLOAT64, device);
}

Tensor::Tensor(const Shape& shape, const std::initializer_list<double> data, const std::string& device):
	Tensor(shape, data.begin(), data.size(), device) {}

Tensor::Tensor(const Shape& shape, const std::initializer_list<float> data, const std::string& device):
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
	t.m_backend = backend::CreateBackendTensor(backend);

	if(!t.m_backend)
		throw std::runtime_error("Could not create tensor with device '" + backend + "'!");
		
	t.dtype = dtype;
	t.shape = shape;
	t.device = backend;

	t->empty(data, ::size(shape), shape, dtype);

	// Reshape to set the strides
	t.reshape_(shape);
	return t;
}

Tensor Tensor::empty(const Shape& shape, const void* data, DTYPE dtype, const std::string& backend)
{
	Tensor t;
	t.m_backend = backend::CreateBackendTensor(backend);

	if(!t.m_backend)
		throw std::runtime_error("Could not create tensor with device '" + backend + "'!");
		
	t.dtype = dtype;
	t.shape = shape;
	t.device = backend;

	t->empty(data, ::size(shape), shape, dtype);

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
	nt.strides = t.strides;
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
	nt.strides = t.strides;
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
	nt.strides = t.strides;
	return nt;
}

size_t Tensor::size() const
{
	return ::size(shape);
}

Tensor Tensor::get(size_t idx) const
{
	AssertExcept(!shape.empty(), "Cannot index an empty tensor!");
	AssertExcept(idx < shape[0], "Index out of bounds error: " << idx << " >= " << shape[0]);

	Tensor t(*this);
	t.shape.erase(t.shape.begin());
	t.strides.erase(t.strides.begin());

	// In case a specific element is requested, ensure the shape is not empty!
	if(t.shape.empty())
	{
		t.strides.push_back(dtypeSizeof(dtype));
		t.shape.push_back(1);
	}

	t.offset += idx * strides[0];
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
	#pragma omp parallel for
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
		case FLOAT32: convertType<From, float>(in, tgt); break;
		case UINT32: convertType<From, uint32_t>(in, tgt); break;
		case INT32: convertType<From, int32_t>(in, tgt); break;
		case FLOAT64: convertType<From, double>(in, tgt); break;
		case UINT64: convertType<From, uint64_t>(in, tgt); break;
		case INT64: convertType<From, int64_t>(in, tgt); break;
		case BYTE: convertType<From, char>(in, tgt); break;
		case UBYTE: convertType<From, unsigned char>(in, tgt); break;
		
		default:
		case NONE: break;
	}
}

Tensor Tensor::astype(DTYPE type) const
{
	// TODO GPU versions?
	Tensor tgt = empty(shape, type);
	Tensor cpuSrc = cpu();

	switch(this->dtype)
	{
		case FLOAT32: convertType<float>(type, *this, tgt); break;
		case UINT32: convertType<uint32_t>(type, *this, tgt); break;
		case INT32: convertType<int32_t>(type, *this, tgt); break;
		case FLOAT64: convertType<double>(type, *this, tgt); break;
		case UINT64: convertType<uint64_t>(type, *this, tgt); break;
		case INT64: convertType<int64_t>(type, *this, tgt); break;
		case BYTE: convertType<char>(type, *this, tgt); break;
		case UBYTE: convertType<unsigned char>(type, *this, tgt); break;
		
		default:
		case NONE: break;
	}

	return tgt.to(device);
}

Tensor Tensor::to(const std::string& device) const
{
	// If we do not change device, return self
	if((device.empty() && m_backend->backendName() == funzel::GetDefaultBackend())
		|| m_backend->backendName() == device)
	{
		return *this;
	}

	Tensor t;
	t.m_backend = backend::CreateBackendTensor(device);

	if(!t.m_backend)
		throw std::runtime_error("Could not create tensor with device '" + device + "'!");

	t.dtype = dtype;
	t.shape = shape;
	t.device = device;
	t.offset = offset;
	t.strides = strides;
	
	t->empty(m_backend->buffer(), m_backend->size, shape, dtype);

	// Reshape to set the strides
	t.reshape_(shape);
	return t;
}

Tensor Tensor::cpu() const
{
	return to("");
}

#include <iostream>
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

	if(!(flags & C_CONTIGUOUS))
	{
		*this = this->unravel();
	}

	this->shape = shape;
	this->strides.resize(shape.size());

	size_t offset = 1;
	for(int i = shape.size() - 1; i >= 0; i--)
	{
		strides[i] = offset*dtypeSizeof(dtype);
		offset *= shape[i];
	}
}

Tensor Tensor::permute(const Shape& shape)
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
		AssertExcept(indices[i] < shape.size(),
			"Cannot permute axes, given index value exceeds the number of dimensions!");

		shape[i] = oldshape[indices[i]];
		strides[i] = oldstrides[indices[i]];
	}
}

static void unravel(Tensor src, Tensor dest)
{
	if(src.shape.empty())
		return;

	#pragma omp parallel for
	for(int64_t i = 0; i < src.shape[0]; i++)
	{
		if(src.shape.size() == 1)
		{
			dest[i] = src[i].item<double>();
		}
		else
		{
			unravel(src[i], dest[i]);
		}
	}
}

Tensor Tensor::unravel() const
{
	// TODO Allow each backend to implement its own ravelling
	//      to enable usage of the GPU!
	Tensor t = Tensor::empty_like(*this);
	::unravel(this->cpu(), t);
	return t.to(this->m_backend->backendName());
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
inline void Apply(const Tensor& a, const Tensor& b, const Tensor& tgt, size_t stopDim, Fn fn, Args&&... args)
{
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

Tensor& Tensor::add_(const Tensor& b, double alpha)
{
	Broadcast<0>(*this, b, *this,
		[](const auto& a, const auto& b) { return b; },
		[](const Tensor& a, Tensor b, Tensor c, double alpha) {
			b->mulAdd(b, c, alpha);
		}, alpha);

	//m_backend->mulAdd(b, *this, alpha);
	return *this;
}

Tensor Tensor::add(const Tensor& b, double alpha) const
{
	Tensor t(*this);
	t.add_(b, alpha);
	return t;
}

Tensor& Tensor::sub_(const Tensor& b, double alpha)
{
	return add_(b, -alpha);
}

Tensor Tensor::sub(const Tensor& b, double alpha) const
{
	Tensor t(*this);
	t.add_(b, -alpha);
	return t;
}

Tensor& Tensor::add_(double alpha)
{
	// TODO Optimized version that does not require a full tensor copy!
	Tensor t = Tensor::empty_like(*this);
	t.fill(alpha);
	m_backend->mulAdd(*this, t, 1.0);

	*this = t;
	return *this;
}

Tensor Tensor::add(double alpha) const
{
	Tensor t(*this);
	t.add_(alpha);
	return t;
}

Tensor& Tensor::mul_(double alpha)
{
	Tensor t = Tensor::zeros_like(*this);
	m_backend->mulAdd(*this, t, alpha);

	*this = std::move(t);
	return *this;
}

Tensor Tensor::mul(double alpha) const
{
	Tensor t = *this;
	t.mul_(alpha);
	return t;
}

Tensor Tensor::div(const Tensor& b) const
{
	Tensor t(*this);
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

#include <iostream>
Tensor Tensor::matmul(const Tensor& b) const
{
	AssertExcept(dtype == b.dtype, "Cannot multiply matrices with different dtypes!");
	Tensor tgt;
	Broadcast<2>(*this, b, tgt,
		[](const auto& a, const auto& b) {
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
# if 0
	

	// std::cout << shape << b.shape << newshape << std::endl;

	Tensor t = Tensor::zeros(newshape, dtype, device);
	m_backend->matmul(*this, b, t);
	return t;
#endif
}

double Tensor::sum()
{
	return m_backend->sum(*this);
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
		case INT32: FillRandom<int32_t>(out.data(out.offset), generator, sz); break;
		case INT64: FillRandom<int64_t>(out.data(out.offset), generator, sz); break;
		case FLOAT32: FillRandom<float>(out.data(out.offset), generator, sz); break;
		case FLOAT64: FillRandom<double>(out.data(out.offset), generator, sz); break;
		case UINT32: FillRandom<uint32_t>(out.data(out.offset), generator, sz); break;
		case UINT64: FillRandom<uint64_t>(out.data(out.offset), generator, sz); break;
		case BYTE: FillRandom<char>(out.data(out.offset), generator, sz); break;
		case UBYTE: FillRandom<unsigned char>(out.data(out.offset), generator, sz); break;
		default: ThrowError("Uknown dtype!");
	}

	return out;
}

Tensor& funzel::randn(Tensor& out)
{
	RandomGenerator<std::uniform_real_distribution<double>> gen{std::uniform_real_distribution<double>(-1.0, 1.0)};
	return randn(out, gen);
}
