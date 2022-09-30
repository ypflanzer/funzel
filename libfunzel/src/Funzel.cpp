#include <funzel/Funzel.hpp>
#include <unordered_map>
#include <string>

#include <iostream>
#include <spdlog/spdlog.h>

using namespace funzel;

static std::unordered_map<std::string, std::unique_ptr<ITensorFactory>> s_tensorBackends;
static const std::string s_defaultBackend = "BLAS";

std::string funzel::GetDefaultBackend()
{
	return s_defaultBackend;
}

void funzel::backend::RegisterTensorBackend(const std::string& name, ITensorFactory* factory)
{
	spdlog::debug("Registering {}", name);
	// std::cout << "Registering " << name << std::endl;
	s_tensorBackends[name] = std::unique_ptr<ITensorFactory>(factory);
}

std::shared_ptr<BackendTensor> funzel::backend::CreateBackendTensor(const std::string& name)
{
	ITensorFactory* factory = nullptr;
	if(name.empty())
	{
		auto iter = s_tensorBackends.find(s_defaultBackend);
		if(iter == s_tensorBackends.end())
			return nullptr;

		return iter->second->create();
	}
	
	// Parse options!
	auto splitterIdx = name.find(":");
	if(splitterIdx == name.npos)
	{
		auto iter = s_tensorBackends.find(s_defaultBackend);
		if(iter == s_tensorBackends.end())
			return nullptr;
		
		return iter->second->create();
	}

	std::string device = name.substr(0, splitterIdx);
	std::string args = name.substr(splitterIdx + 1);

	//std::cout << device << " with " << args << std::endl;

	auto iter = s_tensorBackends.find(device);
	if(iter == s_tensorBackends.end())
		return nullptr;
	
	return iter->second->create(args);
}

static std::vector<DeviceProperties> s_devices;
void funzel::backend::RegisterDevice(const DeviceProperties& props)
{
	s_devices.push_back(props);
}

std::vector<DeviceProperties> funzel::GetDevices()
{
	return s_devices;
}

#include <cmath>
void funzel::PrintDevices()
{
	std::cout << "Found " << s_devices.size() << " device" << (s_devices.size() > 1 ? "s" : "") << ".\n";
	std::cout << "ID\t| Memory \t| Type \t| Name " << std::endl;
	std::cout << std::string(100, '-') << std::endl;

	for(auto& d : s_devices)
	{
		std::string memSizeStr;
		if(d.memorySize == std::numeric_limits<size_t>::max())
		{
			memSizeStr = "invalid";
		}
		else
		{
			const char* sizeUnit = "GB";
			double memSize = d.memorySize / 1024.0f / 1024.0f / 1024.0f;

			if(memSize > 1024)
			{
				memSize /= 1024.0f;
				sizeUnit = "TB";
			}

			memSizeStr = std::to_string(memSize) + sizeUnit;
		}

		std::cout << d.deviceID << "\t| " << memSizeStr << "\t| "
					<< (d.isGPU ? "GPU" : "CPU") << "\t| " << d.deviceName << std::endl;
	}
}


#ifdef WIN32
#include <Windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

void funzel::backend::LoadBackend(const std::string& name)
{
#ifdef WIN32
	auto backend = LoadLibrary(("funzel" + name + ".dll").c_str());
	AssertExcept(backend != NULL, "Could not load backend: " << name);
#elif defined(__unix__) || defined(__APPLE__)
	#ifdef __APPLE__
	const char* ext = ".dylib";
	#else
	const char* ext = ".so";
	#endif

	auto backend = dlopen(("libfunzel" + name + ext).c_str(), RTLD_LAZY);
	AssertExcept(backend != NULL, "Could not load backend: " << name);
#else
	#warning "No implementation for loading backends at runtime was found!"
	AssertExcept(false, "Could not load backend: " << name);
#endif
}

namespace
{
class LogInitializer
{
public:
	LogInitializer()
	{
		spdlog::set_pattern("%^[%=8l]%$ %v");

		auto level = spdlog::level::warn;
		auto* envLevel = getenv("FUNZEL_LOG");
		if(envLevel)
			level = (spdlog::level::level_enum) std::min(std::stoul(envLevel), (ulong) spdlog::level::n_levels);
	
		spdlog::set_level(level);
	}
};

static LogInitializer s_loginit;
}
