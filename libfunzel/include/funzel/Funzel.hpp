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

#include "Type.hpp"

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

#define FUNZEL_INLINE inline

#ifdef _MSC_VER
#define FUNZEL_NOINLINE
#define FUNZEL_RESTRICT __restrict
#else
#define FUNZEL_NOINLINE __attribute__((noinline))
#define FUNZEL_RESTRICT __restrict__
#endif

/**
 * @brief Defines the function signature of a DLL init function.
 */
typedef void (*FunzelInitFuncType)();

/**
 * @fn FUNZEL_REGISTER_BACKEND(name, type)
 * @brief Registers a backend class.
 * @param name The name of the backend.
 * @param type The C++ typename of the backend class.
 */
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
const std::string EmptyStr; ///< An empty string.

/**
 * @brief Throws an exception with the given message.
 * @param msg A message string.
 */
[[noreturn]] inline void ThrowError(const std::string& msg)
{
	throw std::runtime_error(msg);
}

#define STRINGIZE_IMPL(x) #x
#define STRINGIZE(x) STRINGIZE_IMPL(x)

#define UnsupportedOperationError ThrowError("Operation '" + std::string(__func__) + "' is not supported at " __FILE__ ":" STRINGIZE(__LINE__))

/**
 * @fn AssertExcept(v, msg)
 * @brief Thros an exception if the given predicate is false.
 * @param v A truth value.
 * @param msg An error message.
 */
#define AssertExcept(v, msg) if(!(v)) { std::stringstream msgss; msgss << msg; funzel::ThrowError(msgss.str()); }
#endif

class BackendTensor;

/**
 * @brief Provides an interface for a factory producing BackendTensors.
 */
struct ITensorFactory
{
	virtual ~ITensorFactory() = default;

	/**
	 * @brief Creates a new BackendTensor with the given configuration arguments.	 * 
	 * @param args Some implementation defined configuration string.
	 * @return std::shared_ptr<BackendTensor> A new BackendTensor.
	 */
	virtual std::shared_ptr<BackendTensor> create(std::shared_ptr<char> data, size_t count, DTYPE dtype, const std::string& args = EmptyStr) = 0;
	virtual std::shared_ptr<BackendTensor> create(const void* data, size_t count, DTYPE dtype, const std::string& args = EmptyStr) = 0;
};

/**
 * @brief Implements an ITensorFactory for sub-classes of BackendTensor.
 * @tparam Backend The BackendTensor sub-class to return.
 */
template<typename Backend>
struct TensorFactory : public ITensorFactory
{
	std::shared_ptr<BackendTensor> create(std::shared_ptr<char> data, size_t count, DTYPE dtype, const std::string& args = EmptyStr) override
	{
		return Backend::Empty(data, count*dtypeSizeof(dtype), dtype, args);
	}

	std::shared_ptr<BackendTensor> create(const void* data, size_t count, DTYPE dtype, const std::string& args = EmptyStr) override
	{
		return Backend::Empty(data, count*dtypeSizeof(dtype), dtype, args);
	}
};

/**
 * @brief Bundles important information about a general compute device.
 */
struct DeviceProperties
{
	std::string vendorName, deviceName, deviceID;
	uint64_t memorySize; ///< The memory size in bytes.
	bool isGPU;
};

/**
 * @brief Gets all available compute devices.
 * @return The DeviceProperties for all available devices.
 */
FUNZEL_API std::vector<DeviceProperties> GetDevices();

/**
 * @brief Prints a table of available compute devices.
 */
FUNZEL_API void PrintDevices();

/**
 * @brief Get the default backend name string.
 * @return The default string used to configure new BackendTensor's.
 */
FUNZEL_API std::string GetDefaultBackend();

namespace backend
{
/**
 * @brief Registers a new tensor factory.
 * @attention The pointer needs to remain valid for the entire runtime of the program!
 * @param name The name of the backend.
 * @param factory A backend factory object.
 */
FUNZEL_API void RegisterTensorBackend(const std::string& name, ITensorFactory* factory);

#ifndef SWIG
/**
 * @brief Creates a new BackendTensor object given a device configuration string.
 * 
 * The name string is formatted like this: "$DEVICE:$OPTIONS".
 * For example, the first OpenCL device may be used with "OCL:0".
 * See the backend documentation for available options.
 * 
 * @param name The device name and config string.
 * @return A new BackendTensor object.
 */
FUNZEL_API std::shared_ptr<BackendTensor> CreateBackendTensor(std::shared_ptr<char> data, size_t count, DTYPE dtype, const std::string& name);
#endif

/**
 * @brief Creates a new BackendTensor object given a device configuration string.
 * 
 * The name string is formatted like this: "$DEVICE:$OPTIONS".
 * For example, the first OpenCL device may be used with "OCL:0".
 * See the backend documentation for available options.
 * 
 * @param name The device name and config string.
 * @return A new BackendTensor object.
 */
FUNZEL_API std::shared_ptr<BackendTensor> CreateBackendTensor(const void* data, size_t count, DTYPE dtype, const std::string& name);

/**
 * @brief Registers a new device with given properties for use.
 * @param props The propeties of the new device.
 */
FUNZEL_API void RegisterDevice(const DeviceProperties& props);

/**
 * @brief Loads a backend from a shared library.
 * 
 * The right library suffix will be appended automatically for
 * supported platforms (i.e. Linux, macOS and Windows).
 * 
 * @param name The name of the library.
 */
FUNZEL_API void LoadBackend(const std::string& name);

/**
 * @brief Loads a set of default backends.
 * 
 * Uses the FUNZEL_BACKENDS environment variable if it is set
 * as a list of backends delimited by ':' and loads each backend contained
 * therein. Otherwise it scans the directory of the Funzel shared object
 * to find all installed backends.
 */
FUNZEL_API void LoadDefaultBackends();
}

}
