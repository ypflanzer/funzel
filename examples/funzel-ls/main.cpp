#include <funzel/Funzel.hpp>
#include <iostream>

#include <spdlog/spdlog.h>

int main(int /*argc*/, char** /*argv*/)
{
	try
	{
		funzel::backend::LoadBackend("Blas");
		funzel::backend::LoadBackend("OpenCL");
	}
	catch(std::exception& e)
	{
		spdlog::error("Error: {}", e.what());
	}

	funzel::PrintDevices();
	return 0;
}
