#include <funzel/Funzel.hpp>
#include "funzel/OpenCLBackend.hpp"
#include "OpenCLTensor.hpp"

#include <iostream>

using namespace funzel;
using namespace funzel::cl;

FUNZEL_REGISTER_BACKEND("OCL", OpenCLTensor)

static OpenCLBackend s_backend;
static bool s_initialized = false;

OpenCLBackend& OpenCLBackend::the()
{
	return s_backend;
}

void OpenCLBackend::initialize()
{
	if(s_initialized)
		return;

	s_backend.initCL();
	s_initialized = true;
}

void OpenCLBackend::initCL()
{
	try
	{
		std::vector<::cl::Platform> platforms;
		::cl::Platform::get(&platforms);

		std::vector<::cl::Device> devices;
		int ctr = 0;

		for(auto& platform : platforms)
		{
			std::cout << "Using OpenCL Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

			platform.getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);
			std::cout << "Found " << devices.size() << " GPU device(s):" << std::endl;

			// Prefer GPU devices for each platform
			std::sort(devices.begin(), devices.end(), [](const ::cl::Device& a, const ::cl::Device& b) {
				return a.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU;
			});

			::cl::Context context(devices);
			for(auto& d : devices)
			{
				std::cout << '\t' << d.getInfo<CL_DEVICE_VENDOR>() << '\t' << d.getInfo<CL_DEVICE_NAME>() << std::endl;

				DeviceProperties props;
				props.deviceID = "OCL:" + std::to_string(ctr++);
				props.deviceName = d.getInfo<CL_DEVICE_NAME>();
				props.vendorName = d.getInfo<CL_DEVICE_VENDOR>();
				props.memorySize = d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
				props.isGPU = d.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU;
				backend::RegisterDevice(props);

				m_devices.emplace_back(context, d);
			}
		}

		#if 0
		auto m_platform = ::cl::Platform::getDefault();
		std::cout << "Using OpenCL Platform: " << m_platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

		m_platform.getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &m_devices);
		std::cout << "Found " << m_devices.size() << " GPU device(s):" << std::endl;
		
		int ctr = 0;
		for(auto& d : m_devices)
		{
			std::cout << '\t' << d.getInfo<CL_DEVICE_VENDOR>() << '\t' << d.getInfo<CL_DEVICE_NAME>() << std::endl;

			DeviceProperties props;
			props.deviceID = "OCL:" + std::to_string(ctr++);
			props.deviceName = d.getInfo<CL_DEVICE_NAME>();
			props.vendorName = d.getInfo<CL_DEVICE_VENDOR>();
			props.memorySize = d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
			props.isGPU = d.getInfo<CL_DEVICE_TYPE> == CL_DEVICE_TYPE_GPU;
			backend::RegisterDevice(props);
		}

		m_context = ::cl::Context(m_devices);
#endif
		// Initialize cache
		m_cacheDirectory = findCacheDirectory();
		std::cout << "Using cache at: " << m_cacheDirectory << std::endl;
	}
	catch(const ::cl::Error& e)
	{
		std::cout << "Could not initialize OpenCL backend: " << e.err() << std::endl;
	}
}

::cl::CommandQueue OpenCLBackend::getCommandQueue(size_t idx) const
{
	auto& dev = m_devices.at(idx);
	return ::cl::CommandQueue(dev.context, dev.device);
}

CLDevice OpenCLBackend::getDevice(size_t idx) const
{
	return m_devices.at(idx);
}

::cl::Program OpenCLBackend::buildProgram(CLDevice& device, const std::string& src, DTYPE type)
{
	::cl::Program p(device.context, src);
	const auto strtype = dtypeToNativeString(type);
	p.build(device.device, ("-DT=" + strtype + " -DKernel=Kernel_" + strtype).c_str());
	return p;
}

static std::string formatCachedName(const std::string& name, const ::cl::Device& device)
{
	auto cacheName = name + "_" + device.getInfo<CL_DEVICE_VENDOR>() + device.getInfo<CL_DEVICE_NAME>() + device.getInfo<CL_DEVICE_VERSION>() + ".bin";
	
	// Make sure no white-space is in name
	std::replace(cacheName.begin(), cacheName.end(), ' ', '_');
	return cacheName;
}

::cl::Kernel OpenCLBackend::buildKernel(CLDevice& device, const std::string& src, DTYPE type)
{
	::cl::Program p(device.context, src);
	const auto strtype = dtypeToNativeString(type);
	const auto kernelName = "Kernel_" + strtype;
	const auto srcHash = std::hash<std::string>{}(src);

	const auto programName = std::to_string(srcHash) + strtype;
	const auto cacheName = formatCachedName(programName, device.device);

	auto kernel = queryCache(cacheName, device);
	if(kernel.get() != nullptr)
		return kernel;

	//std::cout << "Could not find kernel in cache, building." << std::endl;

	try
	{
		p.build(device.device, ("-DT=" + strtype + " -DKernel=" + kernelName).c_str());
	}
	catch(...)
	{
		cl_int buildErr = CL_SUCCESS;
		auto buildInfo = p.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
		
		for (auto &pair : buildInfo)
		{
			std::cerr << pair.second << std::endl;
		}
	}

	kernel = ::cl::Kernel(p, kernelName.c_str());
	updateCache(cacheName, device.device, p, kernel);
	return kernel;
}

#include <filesystem>
#include <fstream>
#include <iterator>

std::filesystem::path OpenCLBackend::findCacheDirectory()
{
	// First: Try to get the environment variable providing an override.
	{
		char* p = getenv("FUNZEL_CACHE");
		if(p)
		{
			return p;
		}
	}

#if WIN32
	#error "Implement!"
	//CSIDL_LOCAL_APPDATA
	return "";
#else
	char* homePtr = getenv("HOME");
	if(!homePtr) // If no HOME is set, default to CWD
		return std::filesystem::current_path();

	std::filesystem::path cachedir(homePtr);
	cachedir = cachedir / ".cache" / "funzel";

	if(!std::filesystem::exists(cachedir)) // FIXME What to do if it exists but is not a directory?
		std::filesystem::create_directory(cachedir);

	return cachedir;
#endif
}

::cl::Kernel OpenCLBackend::queryCache(const std::string& name, CLDevice device)
{
	auto iter = m_deviceKernels.find(name);
	if(iter != m_deviceKernels.end())
		return iter->second;
	
	const auto inpath = m_cacheDirectory / name;
	if(!std::filesystem::exists(inpath))
		return ::cl::Kernel();

	//std::cout << "Loading binary from: " << inpath << std::endl;
	std::ifstream in(inpath, std::ios::in | std::ios::binary);
	
	AssertExcept(!!in, "Could not open file for reading: " << inpath);
	std::vector<unsigned char> binary((std::istreambuf_iterator<char>(in)), {});

	const unsigned char* binaries[1] = {binary.data()};
	const size_t sz = binary.size();
	int err = 0;
	auto clprog = clCreateProgramWithBinary(device.context(), 1, &device.device(), &sz, binaries, nullptr, &err);
	AssertExcept(err == CL_SUCCESS, "Could not load binary OpenCL program: " << err);

	err = clBuildProgram(clprog, 1, &device.device(), nullptr, nullptr, nullptr);
	AssertExcept(err == CL_SUCCESS, "Could not build binary OpenCL program: " << err);
	
	auto prog = ::cl::Program(clprog, true);
	std::vector<::cl::Kernel> kernels;
	prog.createKernels(&kernels);
	
	AssertExcept(!kernels.empty(), "No kernels were found in binary!");

	m_deviceKernels[name] = kernels[0];
	return kernels[0];
}

void OpenCLBackend::updateCache(const std::string& name, const ::cl::Device& device, ::cl::Program prog, ::cl::Kernel k)
{
	cl_int err;
	auto binaries = prog.getInfo<CL_PROGRAM_BINARIES>(&err);
	AssertExcept(err == CL_SUCCESS, "Could not update OpenCL kernel cache: " << err);
	AssertExcept(!binaries.empty(), "Could not update OpenCL kernel cache, no binaries could be generated: " << err);

	const auto outpath = m_cacheDirectory / name;
	//std::cout << "Saving binary to: " << outpath << std::endl;

	// Write the first valid binary to file.
	// Each program is only built for one specific device, so only one binary will ever be valid.
	for(auto& binary : binaries)
	{
		if(binary.empty())
			continue;

		std::ofstream out(outpath, std::ios::out | std::ios::binary);
		AssertExcept(!!out, "Could not open file for writing: " << outpath);
		out.write((const char*) binary.data(), binary.size());
		break;
	}

	m_deviceKernels[name] = k;
}

void OpenCLBackend::clearCache()
{
	for(auto& e : std::filesystem::directory_iterator(m_cacheDirectory))
	{
		if(e.is_regular_file())
			std::filesystem::remove(e.path());
	}
}

void OpenCLBackend::preloadCache()
{
	for(auto& e : std::filesystem::directory_iterator(m_cacheDirectory))
	{
		//if(e.is_regular_file())
			
	}
}
