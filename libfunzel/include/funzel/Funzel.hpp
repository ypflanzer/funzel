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

namespace funzel
{

const std::string EmptyStr;

[[noreturn]] inline void ThrowError(const std::string& msg)
{
	throw std::runtime_error(msg);
}

#define AssertExcept(v, msg) if(!(v)) { std::stringstream msgss; msgss << msg; funzel::ThrowError(msgss.str()); }

#define FUNZEL_REGISTER_BACKEND(name, type)\
struct StaticInitializer##type \
{\
	using T = type; \
	StaticInitializer##type() { backend::RegisterTensorBackend(name, new TensorFactory<T>); T::initializeBackend(); } \
};\
volatile static StaticInitializer##type s_initializer;

class BackendTensor;
struct ITensorFactory
{
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

std::vector<DeviceProperties> GetDevices();
void PrintDevices();
std::string GetDefaultBackend();

namespace backend
{
void RegisterTensorBackend(const std::string& name, ITensorFactory* factory);
std::shared_ptr<BackendTensor> CreateBackendTensor(const std::string& name);
void RegisterDevice(const DeviceProperties& props);
}

}
