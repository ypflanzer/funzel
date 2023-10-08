#define NOMINMAX

#include <funzel/Funzel.hpp>
#include <unordered_map>
#include <string>

#include <cmath>
#include <iostream>
#include <filesystem>
#include <spdlog/spdlog.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

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

template<typename T>
static inline std::shared_ptr<BackendTensor> CreateBackendTensorT(T data, size_t count, DTYPE dtype, const std::string& name)
{
	// If no backend is available, try loading the default backends before
	// continuing.
	if(s_tensorBackends.empty())
	{
		backend::LoadDefaultBackends();
	}

 	if(name.empty())
	{
		auto iter = s_tensorBackends.find(s_defaultBackend);
		if(iter == s_tensorBackends.end())
			return nullptr;

		return iter->second->create(data, count, dtype);
	}
	
	// Parse options!
	auto splitterIdx = name.find(":");
	if(splitterIdx == name.npos)
	{
		auto iter = s_tensorBackends.find(s_defaultBackend);
		if(iter == s_tensorBackends.end())
			return nullptr;
		
		return iter->second->create(data, count, dtype);
	}

	std::string device = name.substr(0, splitterIdx);
	std::string args = name.substr(splitterIdx + 1);

	//std::cout << device << " with " << args << std::endl;

	auto iter = s_tensorBackends.find(device);
	if(iter == s_tensorBackends.end())
		return nullptr;
	
	return iter->second->create(data, count, dtype, args);
}

std::shared_ptr<BackendTensor> funzel::backend::CreateBackendTensor(std::shared_ptr<char> data, size_t count, DTYPE dtype, const std::string& name)
{
	return CreateBackendTensorT(data, count, dtype, name);
}

std::shared_ptr<BackendTensor> funzel::backend::CreateBackendTensor(const void* data, size_t count, DTYPE dtype, const std::string& name)
{
	return CreateBackendTensorT(data, count, dtype, name);
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

void LoadBackendFile(const std::string& filename)
{
	spdlog::debug("Loading backend: {}", filename);
#ifdef _WIN32
	auto backend = LoadLibrary(filename.c_str());
	AssertExcept(backend != NULL, "Could not load backend: " << name);

	auto InitFunc = (FunzelInitFuncType) GetProcAddress(backend, "FunzelInit");
	if(InitFunc != NULL)
		InitFunc();

#elif defined(__unix__) || defined(__APPLE__)
	auto backend = dlopen(filename.c_str(), RTLD_LAZY);
	AssertExcept(backend != NULL, "Could not load backend '" << filename << "': " << dlerror());

	auto InitFunc = (FunzelInitFuncType) dlsym(backend, "FunzelInit");
	if(InitFunc != NULL)
		InitFunc();
#else
	#warning "No implementation for loading backends at runtime was found!"
	AssertExcept(false, "Could not load backend: " << name);
#endif
}

void funzel::backend::LoadBackend(const std::string& name)
{
	#ifdef _WIN32
		LoadBackendFile("funzel" + name + ".dll");
	#else
		#ifdef __APPLE__
		const char* ext = ".dylib";
		#else
		const char* ext = ".so";
		#endif

		LoadBackendFile("libfunzel" + name + ext);
	#endif
}

void funzel::backend::LoadDefaultBackends()
{
	char* backendsEnv = getenv("FUNZEL_BACKENDS");
	if(backendsEnv)
	{
		spdlog::debug("Loading backends from environment variable.");
		std::stringstream backends(backendsEnv);
		std::string entry;
		while(std::getline(backends, entry, ':'))
		{
			LoadBackend(entry);
		}
	}
	else
	{
		spdlog::debug("Loading backends from installation directory.");
		std::filesystem::path soPath;
		#ifndef _WIN32
		#ifdef __APPLE__
			constexpr const char* ext = ".dylib";
		#else
			constexpr const char* ext = ".so";
		#endif
			constexpr const char* backendPrefix = "libfunzel";

			Dl_info info;
			if(!dladdr((void*) funzel::backend::LoadDefaultBackends, &info))
				throw std::runtime_error("Could not query installed backends!");

			soPath = std::filesystem::path(info.dli_fname).replace_filename("");
		#else
			constexpr const char* ext = ".dll";
			constexpr const char* backendPrefix = "funzel";

			TCHAR moduleFileName[MAX_PATH];
			if(!GetModuleFileName(NULL, moduleFileName, MAX_PATH))
				throw std::runtime_error("Could not query installed backends!");
			
			soPath = moduleFileName;
			soPath.replace_filename("");
		#endif

		spdlog::debug("Searching for backends in: {}", soPath.string());
		for(const auto& e : std::filesystem::directory_iterator(soPath))
		{
			if(!e.is_regular_file())
				continue;

			auto path = e.path();
			if(path.extension() != ext)
				continue;

			//path.replace_extension("");

			auto filename = path.filename().string();
			if(filename.find(backendPrefix) != 0)
				continue;
			
			LoadBackendFile(path.string());
		}
	}
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
			level = (spdlog::level::level_enum) std::min(std::stoul(envLevel), (unsigned long) spdlog::level::n_levels);
	
		spdlog::set_level(level);
	}
};

static LogInitializer s_loginit;
}
