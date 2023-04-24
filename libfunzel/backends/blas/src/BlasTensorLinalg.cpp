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
#define NOMINMAX
#include <funzel/Tensor.hpp>
#include "BlasTensor.hpp"

#include <iostream>
#include <spdlog/spdlog.h>

#define LAPACK_COMPLEX_STRUCTURE
#define HAVE_LAPACK_CONFIG_H
#include <lapacke.h>

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

template<typename T>
inline static void DoTrace(const Tensor& self, Tensor& tgt)
{
	T traceval = 0;
	for(size_t i = 0; i < self.shape[0]; i++)
	{
		traceval += self[{i, i}].item<T>();
	}
	tgt.set(traceval);
}

void BlasTensor::trace(const Tensor& self, Tensor tgt)
{
	if(self.shape.empty())
		return;
	
	if(self.shape.size() > 2)
	{
		//#pragma omp parallel for
		for(int i = 0; i < self.shape[0]; i++)
		{
			trace(self[i], tgt[i]);
		}

		return;
	}

	AssertExcept(self.shape[0] == self.shape[1], "Calculating the trace requires a square matrix.");

	switch(dtype)
	{
		case DFLOAT32: {
			DoTrace<float>(self, tgt);
		} break;
		case DFLOAT64: {
			DoTrace<double>(self, tgt);
		} break;
		default: ThrowError("Unsupported dtype!");
	}
}

void BlasTensor::svd(const Tensor& self, Tensor U, Tensor S, Tensor V)
{
	if(self.shape.empty())
		return;
	
	if(self.shape.size() > 2)
	{
		//#pragma omp parallel for
		for(int i = 0; i < self.shape[0]; i++)
		{
			svd(self[i], U[i], S[i], V[i]);
		}

		return;
	}

	//AssertExcept(self.shape[0] == self.shape[1], "Calculating the trace requires a square matrix.");
	int n = self.shape[0], m = self.shape[1];
	void* udata = U.data(U.offset);
	void* sdata = S.data(S.offset);
	void* vdata = V.data(V.offset);
	const void* selfdata = self.data(self.offset);

	int errcode = 0;
	switch(dtype)
	{
		case DFLOAT32: {
			errcode = LAPACKE_sgesdd(LAPACK_ROW_MAJOR, 'A', m, n,
				(float*) selfdata, m,
				(float*) sdata,
				(float*) udata, m, // TODO Strides!
				(float*) vdata, n);
		} break;
		case DFLOAT64: {

		} break;
		default: ThrowError("Unsupported dtype!");
	}

	spdlog::info("Errcode: {}", errcode);
	//SGESDD
	//sgesdd_();

	//LAPACKE_sgesdd(LAPACK_ROW_MAJOR, 'A', m, n, self.data(), m, s.data(), u.data());
}
