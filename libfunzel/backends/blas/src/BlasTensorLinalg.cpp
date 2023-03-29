#define NOMINMAX
#include <funzel/Tensor.hpp>
#include "BlasTensor.hpp"

#include <lapacke.h>

#include <iostream>

using namespace funzel;
using namespace blas;

template<typename T>
inline static void DoDet(Tensor& luMat, Tensor& tgt)
{
	const void* src = luMat.data(luMat.offset);

	int n = luMat.shape[0], m = luMat.shape[1];
	int info = 1;

	// FIXME: Maybe for smaller data we can use the stack? See alloca and malloca
	const auto numPivots = std::min(m, n);
	auto pivots = std::make_unique<int[]>(numPivots);

	if constexpr(std::is_same_v<T, float>)
	{
		sgetrf_(&m, &n, (float*) src, &m, pivots.get(), &info);
	}
	else if constexpr(std::is_same_v<T, float>)
	{
		dgetrf_(&m, &n, (double*) src, &m, pivots.get(), &info);
	}
	else
	{

	}

	// If the matrix is singular.
	if(info != 0)
	{
		tgt.set(0);
		return;
	}

	T detval = 1;
	for(size_t i = 0; i < numPivots; i++)
	{
		detval *= luMat[{i, i}].item<T>();
		if(pivots[i] != i + 1) // FIXME: Is this right?
			detval *= T(-1);
	}

	tgt.set(detval);
}

void BlasTensor::det(const Tensor& self, Tensor tgt)
{
	if(self.shape.empty())
		return;
	
	if(self.shape.size() > 2)
	{
		//#pragma omp parallel for
		for(int i = 0; i < self.shape[0]; i++)
		{
			det(self[i], tgt[i]);
		}

		return;
	}

	auto luMat = self.clone();

	switch(dtype)
	{
		case DFLOAT32: {
			DoDet<float>(luMat, tgt);
		} break;
		case DFLOAT64: {
			DoDet<double>(luMat, tgt);
		} break;
		default: ThrowError("Unsupported dtype!");
	}
}

void BlasTensor::inv(const Tensor& self, Tensor tgt)
{
	if(self.shape.empty())
		return;
	
	if(self.shape.size() > 2)
	{
		//#pragma omp parallel for
		for(int i = 0; i < self.shape[0]; i++)
		{
			inv(self[i], tgt[i]);
		}

		return;
	}

	tgt.set(self);

	const void* src = tgt.data(tgt.offset);
	int n = tgt.shape[0], m = tgt.shape[1], lwork = n*n;
	int info = 1;

	// FIXME: Maybe for smaller data we can use the stack? See alloca and malloca
	const auto numPivots = std::min(m, n);
	auto pivots = std::make_unique<int[]>(numPivots);

	switch(dtype)
	{
		case DFLOAT32: {
			sgetrf_(&m, &n, (float*) src, &m, pivots.get(), &info);
			AssertExcept(info == 0, "Could not get LU decomposition: " << info);

			auto work = std::make_unique<float>(lwork);
			sgetri_(&n, (float*) src, &m, pivots.get(), work.get(), &lwork, &info);
			AssertExcept(info == 0, "Could not invert matrix: " << info);
		} break;
		case DFLOAT64: {
			dgetrf_(&m, &n, (double*) src, &m, pivots.get(), &info);
			AssertExcept(info == 0, "Could not get LU decomposition: " << info);

			auto work = std::make_unique<double>(lwork);
			dgetri_(&n, (double*) src, &m, pivots.get(), work.get(), &lwork, &info);
			AssertExcept(info == 0, "Could not invert matrix: " << info);
		} break;
		default: ThrowError("Unsupported dtype!");
	}
}
