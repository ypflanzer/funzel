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

#include <exception>
#include <memory>
#include <string>
#include <vector>
#include <sstream>

// Definitions for symbol export
#ifdef WIN32
	#define EXPORT __declspec(dllexport)
#else
	#define EXPORT __attribute__((visibility("default")))
#endif

#ifdef FUNZEL_EXPORT
	#define FUNZEL_API EXPORT
#else
	#ifdef WIN32
		#define FUNZEL_API __declspec(dllimport)
	#else
		#define FUNZEL_API
	#endif
#endif

#define FUNZEL_REGISTER_BACKEND(name, type)\
struct EXPORT StaticInitializer##type \
{\
	using T = type; \
	StaticInitializer##type() { backend::RegisterTensorBackend(name, new TensorFactory<T>); T::initializeBackend(); } \
};\
volatile static StaticInitializer##type s_initializer;

namespace funzel
{

#ifndef SWIG
const std::string EmptyStr;
[[noreturn]] inline void ThrowError(const std::string& msg)
{
	throw std::runtime_error(msg);
}

#define AssertExcept(v, msg) if(!(v)) { std::stringstream msgss; msgss << msg; funzel::ThrowError(msgss.str()); }
#endif

class BackendTensor;
struct ITensorFactory
{
	virtual ~ITensorFactory() = default;
	virtual std::shared_ptr<BackendTensor> create(const std::string& args = EmptyStr) = 0;
};

template<typename Backend>
struct TensorFactory : public ITensorFactory
{
	std::shared_ptr<BackendTensor> create(const std::string& args = EmptyStr) override
	{
		return std::make_shared<Backend>(args);
	}
};

struct DeviceProperties
{
	std::string vendorName, deviceName, deviceID;
	uint64_t memorySize;
	bool isGPU;
};

FUNZEL_API std::vector<DeviceProperties> GetDevices();
FUNZEL_API void PrintDevices();
FUNZEL_API std::string GetDefaultBackend();

namespace backend
{
FUNZEL_API void RegisterTensorBackend(const std::string& name, ITensorFactory* factory);
FUNZEL_API std::shared_ptr<BackendTensor> CreateBackendTensor(const std::string& name);
FUNZEL_API void RegisterDevice(const DeviceProperties& props);

FUNZEL_API void LoadBackend(const std::string& name);
}

}
