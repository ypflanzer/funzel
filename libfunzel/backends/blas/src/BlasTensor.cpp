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

#include <cstring>
#include <spdlog/spdlog.h>
#include <stdexcept>

#include "BlasTensorImpl.hpp"

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

static inline void* checkedMalloc(size_t sz)
{
	void* p = nullptr;

#if 1
	p = std::malloc(sz);
#else
	p = std::aligned_alloc(0x40, sz);
#endif

	if(!p)
		// Don't use std::bad_alloc because this is only used to allocate Tensors which may be very large and
		// fail because of that such that normal operation can continue afterwards.
		throw std::runtime_error("Could not allocate " + std::to_string(sz) + " bytes of memory: " + std::strerror(errno));
	return p;
}

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
	AssertExcept(m_data, "Tensor data is null!");
	return m_data.get() + offset;
}

template<SIMD_TYPE SimdType = SIMD_TYPE::NONE>
static inline BlasTensor* CreateBlasTensor(DTYPE dtype)
{
	switch(dtype)
	{
		case DFLOAT32: return new BlasTensorImpl<float, SimdType>(); break;
		case DFLOAT64: return new BlasTensorImpl<double, SimdType>(); break;
		case DINT8: return new BlasTensorImpl<int8_t, SimdType>(); break;
		case DINT16: return new BlasTensorImpl<int16_t, SimdType>(); break;
		case DINT32: return new BlasTensorImpl<int32_t, SimdType>(); break;
		case DINT64: return new BlasTensorImpl<int64_t, SimdType>(); break;
		case DUINT8: return new BlasTensorImpl<uint8_t, SimdType>(); break;
		case DUINT16: return new BlasTensorImpl<uint16_t, SimdType>(); break;
		case DUINT32: return new BlasTensorImpl<uint32_t, SimdType>(); break;
		case DUINT64: return new BlasTensorImpl<uint64_t, SimdType>(); break;

		default: throw std::invalid_argument("Unsupported DTYPE given: " + std::to_string(dtype));
	}

	// Should never happen!
	return nullptr;
}

static inline BlasTensor* CreateBlasTensor(DTYPE dtype, const std::string& simd)
{
	if(simd == "AVX2")
		return CreateBlasTensor<SIMD_TYPE::AVX2>(dtype);
	else
		return CreateBlasTensor<SIMD_TYPE::NONE>(dtype);
}

std::shared_ptr<BackendTensor> BlasTensor::Empty(std::shared_ptr<char> data, size_t sz, DTYPE dtype, const std::string& args)
{
	BlasTensor* tensor = CreateBlasTensor(dtype, args);
	tensor->empty(data, sz);
	return std::shared_ptr<BackendTensor>(tensor);
}

std::shared_ptr<BackendTensor> BlasTensor::Empty(const void* data, size_t sz, DTYPE dtype, const std::string& args)
{
	BlasTensor* tensor = CreateBlasTensor(dtype, args);
	tensor->empty(data, sz);
	return std::shared_ptr<BackendTensor>(tensor);
}

void BlasTensor::empty(std::shared_ptr<char> buffer, size_t sz)
{
	this->size = sz;
	if(!buffer)
	{
		m_data = std::shared_ptr<char>((char*) checkedMalloc(sz));
	}
	else
	{
		m_data = buffer;
	}
}

void BlasTensor::empty(const void* buffer, size_t sz)
{
	this->size = sz;
	m_data = std::shared_ptr<char>((char*) checkedMalloc(sz));

	if(buffer)
	{
		memcpy(m_data.get(), buffer, sz);
	}
}

std::shared_ptr<BackendTensor> BlasTensor::clone() const
{
	BlasTensor* t = CreateBlasTensor(dtype);
	auto data = std::shared_ptr<char>((char*) checkedMalloc(size));
	memcpy(data.get(), m_data.get(), size);

	t->empty(data, size);
	return std::shared_ptr<BackendTensor>(t);
}
