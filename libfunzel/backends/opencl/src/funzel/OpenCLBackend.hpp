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

#define CL_HPP_ENABLE_EXCEPTIONS

#if defined(__APPLE__) || defined(WIN32)
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include "opencl.hpp"
#else
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>
#endif

#include <string>
#include <filesystem>
#include <unordered_map>

#include <funzel/Tensor.hpp>

namespace funzel
{
namespace cl
{

struct CLDevice
{
	CLDevice() = default;
	CLDevice(const ::cl::Context& ctx, const ::cl::Device& dev):
		device(dev),
		context(ctx),
		queue(::cl::CommandQueue(ctx, dev)) {}
	
	::cl::Device device;
	::cl::Context context;
	::cl::CommandQueue queue;
};

class OpenCLBackend
{
public:
	static OpenCLBackend& the();
	static void initialize();

	OpenCLBackend() { initCL(); }

	void initCL();

	//::cl::Context& getContext() { return m_context; }
	::cl::CommandQueue getCommandQueue(size_t idx) const;
	CLDevice getDevice(size_t idx) const;

	size_t parseArgs(const std::string& str) const;

	void setBuildOptions(const std::string& opts)
	{
		m_buildOptions = opts;
	}

	void addBuildOptions(const std::string& opts)
	{
		m_buildOptions += opts;
	}

	std::string getBuildOptions() const { return m_buildOptions; }


	::cl::Program buildProgram(CLDevice& device, const std::string& src, DTYPE type);
	::cl::Kernel buildKernel(CLDevice& device, const std::string& src, DTYPE type);

	//size_t getLocalWorkgroup() const { return m_maxLocalWorkgroup; }

	::cl::Kernel queryCache(const std::string& name, CLDevice device);
	void updateCache(const std::string& name, const ::cl::Device& device, ::cl::Program prog, ::cl::Kernel k);
	void clearCache();
	void preloadCache();

	std::filesystem::path getCacheDirectory() const;

	static std::filesystem::path findCacheDirectory();

private:
	std::filesystem::path m_cacheDirectory;
	std::unordered_map<std::string, ::cl::Kernel> m_deviceKernels;
	
	std::vector<CLDevice> m_devices;
	std::string m_buildOptions = " -cl-mad-enable -cl-std=CL1.2 ";
};

}
}