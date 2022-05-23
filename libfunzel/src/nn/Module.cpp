#include <funzel/nn/Module.hpp>

using namespace funzel;
using namespace nn;

void Module::defaultInitialize()
{
	for(auto& p : m_parameters)
	{
		randn(p);
	}
}

void Module::to(const std::string& device)
{
	for(auto& p : m_parameters)
		p = p.to(device);
}
