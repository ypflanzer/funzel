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
#define NOMINMAX
#include <funzel/Tensor.hpp>
#include "BlasTensor.hpp"

#include <iostream>
#include <cstring>
#include <cmath>
#include <functional>
#include <spdlog/spdlog.h>

#include "BlasTensorImpl.hpp"

#include <cblas.h>

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

#ifdef BLAS_VENDOR_OPENBLAS
	char* corename = openblas_get_corename();
	spdlog::debug("OpenBLAS Corename (may not be accurate): {}", corename);
#endif
}

void* BlasTensor::data(size_t offset)
{
	// TODO: Bounds check!
	const auto sz = size*dtypeSizeof(dtype);
	AssertExcept(offset < sz, "Out of bounds access: " + std::to_string(offset) + " >= " + std::to_string(sz));
	return m_data.get() + offset;
}

static inline BlasTensor* CreateBlasTensor(DTYPE dtype)
{
	switch(dtype)
	{
		default:
		case DFLOAT32: return new BlasTensorImpl<float>(); break;

		case DFLOAT64: return new BlasTensorImpl<double>(); break;
		case DINT8: return new BlasTensorImpl<int8_t>(); break;
		case DINT16: return new BlasTensorImpl<int16_t>(); break;
		case DINT32: return new BlasTensorImpl<int32_t>(); break;
		case DINT64: return new BlasTensorImpl<int64_t>(); break;
		case DUINT8: return new BlasTensorImpl<uint8_t>(); break;
		case DUINT16: return new BlasTensorImpl<uint16_t>(); break;
		case DUINT32: return new BlasTensorImpl<uint32_t>(); break;
	}

	// Should never happen!
	return nullptr;
}

std::shared_ptr<BackendTensor> BlasTensor::Empty(std::shared_ptr<char> data, size_t sz, DTYPE dtype, const std::string& args)
{
	BlasTensor* tensor = CreateBlasTensor(dtype);
	tensor->empty(data, sz);
	return std::shared_ptr<BackendTensor>(tensor);
}

std::shared_ptr<BackendTensor> BlasTensor::Empty(const void* data, size_t sz, DTYPE dtype, const std::string& args)
{
	BlasTensor* tensor = CreateBlasTensor(dtype);
	tensor->empty(data, sz);
	return std::shared_ptr<BackendTensor>(tensor);
}

void BlasTensor::empty(std::shared_ptr<char> buffer, size_t sz)
{
	this->size = sz;
	if(!buffer)
	{
		m_data = std::shared_ptr<char>((char*) std::malloc(size*dtypeSizeof(dtype)));
	}
	else
	{
		m_data = buffer;
	}
}

void BlasTensor::empty(const void* buffer, size_t sz)
{
	this->size = sz;

	m_data = std::shared_ptr<char>((char*) std::malloc(size*dtypeSizeof(dtype)));

	if(buffer)
	{
		memcpy(m_data.get(), buffer, this->size*dtypeSizeof(dtype));
	}
}

void BlasTensor::empty(std::shared_ptr<char> buffer, size_t sz, DTYPE dtype)
{
	this->size = sz;
	this->dtype = dtype;

	if(!buffer)
	{
		// m_data = std::shared_ptr<char>((char*) std::aligned_alloc(16, size*dtypeSizeof(dtype)));
		m_data = std::shared_ptr<char>((char*) std::malloc(size*dtypeSizeof(dtype)));
	}
	else
	{
		m_data = buffer;
	}
}

void BlasTensor::empty(const void* buffer, size_t sz, DTYPE dtype)
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
	BlasTensor* t = CreateBlasTensor(dtype);
	size_t sz = size*dtypeSizeof(dtype);
	auto data = std::shared_ptr<char>((char*) std::malloc(sz));

	memcpy(data.get(), m_data.get(), sz);

	t->empty(data, sz);
	return std::shared_ptr<BackendTensor>(t);
}

static void CopyTensor(Tensor src, Tensor dest)
{
	if(src.shape.empty())
		return;

	if(src.shape.size() == 1)
	{
		// #pragma omp parallel for
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

template<typename T, typename Fn>
inline void TensorOpInner(const Tensor& self, Tensor& tgt, Fn op)
{
	for(int64_t x = 0; x < self.shape[0]; x++)
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

template<bool EnableUnsigned = true, typename Fn>
inline void TensorOp(const Tensor& self, Tensor& tgt, Fn op)
{
	if constexpr (EnableUnsigned)
	{
		switch(self.dtype)
		{
			case DUINT16: {
				TensorOpOuter<uint16_t>(self, tgt, op);
				return;
			}
			case DUINT32: {
				TensorOpOuter<uint32_t>(self, tgt, op);
				return;
			}
			case DUINT64: {
				TensorOpOuter<uint64_t>(self, tgt, op);
				return;
			}

			default: {}
		}
	}

	switch(self.dtype)
	{
		case DFLOAT32: {
			TensorOpOuter<float>(self, tgt, op);
		} break;
		case DFLOAT64: {
			TensorOpOuter<double>(self, tgt, op);
		} break;
		
		case DINT8: {
			TensorOpOuter<int8_t>(self, tgt, op);
		} break;
		case DUINT8: {
			TensorOpOuter<uint8_t>(self, tgt, op);
		} break;
		
		case DINT16: {
			TensorOpOuter<int16_t>(self, tgt, op);
		} break;
		case DINT32: {
			TensorOpOuter<int32_t>(self, tgt, op);
		} break;
		case DINT64: {
			TensorOpOuter<int64_t>(self, tgt, op);
		} break;
		default: ThrowError("Unsupported dtype!");
	}
}

void BlasTensor::unravel(const Tensor& self, Tensor tgt)
{
	TensorOp(self, tgt, [](const auto& v) { return v; });
}
