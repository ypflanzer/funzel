#include <funzel/nn/Sequential.hpp>

using namespace funzel;
using namespace nn;

Sequential::Sequential(std::initializer_list<ModuleRef> modules):
	m_modules(modules) {}

Tensor Sequential::forward(const Tensor& input)
{
	Tensor res = input;
	for(auto& m : m_modules)
		res = m->forward(res);

	return res;
}

Tensor Sequential::backward(const Tensor& /*input*/)
{
	return Tensor();
}

void Sequential::to(const std::string& device)
{
	for(auto& m : m_modules)
		m->to(device);
}

void Sequential::defaultInitialize()
{
	for(auto& m : m_modules)
		m->defaultInitialize();
}
