/* 
 * This file is part of Funzel.
 * Copyright (c) 2022-2023 Yannick Pflanzer.
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

#include "funzel/Funzel.hpp"
#include <lapacke.h>

namespace funzel
{
namespace blas
{

template<typename T>
inline static void DoDet(Tensor& luMat, Tensor& tgt)
{
	const void* src = luMat.data(luMat.offset);

	lapack_int n = luMat.shape[0], m = luMat.shape[1];
	lapack_int info = 1;

	// FIXME: Maybe for smaller data we can use the stack? See alloca and malloca
	const auto numPivots = std::min(m, n);
	auto pivots = std::make_unique<lapack_int[]>(numPivots);

	if constexpr(std::is_same_v<T, float>)
	{
		//sgetrf_(&m, &n, (float*) src, &m, pivots.get(), &info);
		info = LAPACKE_sgetrf(LAPACK_COL_MAJOR, m, n, (float*) src, n, pivots.get());
	}
	else if constexpr(std::is_same_v<T, float>)
	{
		//dgetrf_(&m, &n, (double*) src, &m, pivots.get(), &info);
		info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, m, n, (double*) src, n, pivots.get());
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
	for(int64_t i = 0; i < numPivots; i++)
	{
		detval *= luMat[{i, i}].item<T>();
		if(pivots[i] != i + 1) // FIXME: Is this right?
			detval *= T(-1);
	}

	tgt.set(detval);
}

template<typename T, SIMD_TYPE SimdType>
void BlasTensorImpl<T, SimdType>::det(const Tensor& self, Tensor tgt)
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
	DoDet<T>(luMat, tgt);
}

template<typename T, SIMD_TYPE SimdType>
void BlasTensorImpl<T, SimdType>::inv(const Tensor& self, Tensor tgt)
{
	if(self.shape.empty())
		return;
	
	if(self.shape.size() > 2)
	{
		//#pragma omp parallel for
		for(int64_t i = 0; i < self.shape[0]; i++)
		{
			inv(self[i], tgt[i]);
		}

		return;
	}

	tgt.set(self);

	const void* src = tgt.data(tgt.offset);
	lapack_int n = tgt.shape[0], m = tgt.shape[1];//, lwork = n*n;
	lapack_int info = 1;

	// FIXME: Maybe for smaller data we can use the stack? See alloca and malloca
	const auto numPivots = std::min(m, n);
	auto pivots = std::make_unique<lapack_int[]>(numPivots);

	if constexpr (std::is_same_v<T, float>)
	{
		//sgetrf_(&m, &n, (float*) src, &m, pivots.get(), &info);
		info = LAPACKE_sgetrf(LAPACK_COL_MAJOR, m, n, (float*) src, n, pivots.get());
		AssertExcept(info == 0, "Could not get LU decomposition: " << info);

		//auto work = std::make_unique<float>(lwork);
		//sgetri_(&n, (float*) src, &m, pivots.get(), work.get(), &lwork, &info);
		LAPACKE_sgetri(LAPACK_COL_MAJOR, n, (float*) src, m, pivots.get());
		AssertExcept(info == 0, "Could not invert matrix: " << info);
	}
	else if constexpr (std::is_same_v<T, double>)
	{
		//dgetrf_(&m, &n, (double*) src, &m, pivots.get(), &info);
		info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, m, n, (double*) src, n, pivots.get());
		AssertExcept(info == 0, "Could not get LU decomposition: " << info);

		//auto work = std::make_unique<double>(lwork);
		//dgetri_(&n, (double*) src, &m, pivots.get(), work.get(), &lwork, &info);
		LAPACKE_dgetri(LAPACK_COL_MAJOR, n, (double*) src, m, pivots.get());
		AssertExcept(info == 0, "Could not invert matrix: " << info);
	}
	else
	{
		ThrowError("Unsupported dtype!");
	}
}

template<typename T>
inline static void DoTrace(const Tensor& self, Tensor& tgt)
{
	T traceval = 0;

	for(int64_t i = 0; i < self.shape[0]; i++)
	{
		traceval += self[{i, i}].item<T>();
	}
	tgt.set(traceval);
}

template<typename T, SIMD_TYPE SimdType>
void BlasTensorImpl<T, SimdType>::trace(const Tensor& self, Tensor tgt)
{
	if(self.shape.empty())
		return;
	
	if(self.shape.size() > 2)
	{
		//#pragma omp parallel for
		for(int64_t i = 0; i < self.shape[0]; i++)
		{
			trace(self[i], tgt[i]);
		}

		return;
	}

	AssertExcept(self.shape[0] == self.shape[1], "Calculating the trace requires a square matrix.");
	DoTrace<T>(self, tgt);
}

template<typename T, SIMD_TYPE SimdType>
void BlasTensorImpl<T, SimdType>::svd(const Tensor& self, Tensor U, Tensor S, Tensor V, bool fullMatrices)
{
	if(self.shape.empty())
		return;
	
	if(self.shape.size() > 2)
	{
		//#pragma omp parallel for
		for(int64_t i = 0; i < self.shape[0]; i++)
		{
			svd(self[i], U[i], S[i], V[i]);
		}

		return;
	}

	lapack_int m = self.shape[0], n = self.shape[1];
	void* udata = U.data(U.offset);
	void* sdata = S.data(S.offset);
	void* vdata = V.data(V.offset);
	const void* selfdata = self.data(self.offset);

	AssertExcept(udata && sdata && vdata && selfdata, "All tensors must be allocated!");

	const int mode = LAPACK_COL_MAJOR; //(self.isContiguous() ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR);
	const char job = fullMatrices ? 'A' : 'S';

	int errcode = 0;
	if constexpr (std::is_same_v<T, float>)
	{
		errcode = LAPACKE_sgesdd(mode, job, m, n,
				(float*) selfdata, m,
				(float*) sdata,
				(float*) udata, m, // TODO Strides!
				(float*) vdata, n);
	}
	else if constexpr (std::is_same_v<T, double>)
	{
		errcode = LAPACKE_dgesdd(mode, job, m, n,
				(double*) selfdata, m,
				(double*) sdata,
				(double*) udata, m, // TODO Strides!
				(double*) vdata, n);
	}
	else
	{
		ThrowError("Unsupported dtype!");
	}
	
	AssertExcept(errcode == 0, "Error running SVD: " << errcode);
}

}
}
