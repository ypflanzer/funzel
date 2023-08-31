#pragma once

#include "BlasTensor.hpp"

namespace funzel
{
namespace blas
{

template<typename T>
class EXPORT BlasTensorImpl: public BlasTensor
{
public:
	BlasTensorImpl()
	{
		dtype = funzel::dtype<T>();
	}

	void fill(const Tensor& self, double scalar) override
	{
		const auto sz = self.size();
		auto* d = reinterpret_cast<T*>(data(self.offset));
		
		// TODO Make configurable!
		#pragma omp parallel for if(sz > 4096)
		for(int64_t i = 0; i < sz; i++)
		{
			d[i] = scalar;
		}
	}
};

}
}
