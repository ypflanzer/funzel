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

#if !defined(FUNZEL_NO_SIMD)
	#if defined(__x86_64__) || defined(_M_X64)
	#include <immintrin.h>
	#elif defined(__aarch64__) || defined(_M_ARM64)
	#endif
#endif

namespace funzel
{
namespace blas
{

template<typename T, SIMD_TYPE SimdType>
template<typename AccumT>
AccumT BlasTensorImpl<T, SimdType>::sum(const T* FUNZEL_RESTRICT data, size_t n, size_t stride)
{
	AccumT accum(0);
	for(size_t off = 0; off < n*stride; off += stride)
	{
		accum += data[off];
	}

	return accum;
}

// SIMD optimized kernels start here.
#ifndef FUNZEL_NO_SIMD

#ifdef __AVX2__

static inline float HorizontalAddAVX2_ps(__m256 vector)
{
	const __m256 sum1 = _mm256_hadd_ps(vector, vector);
	const __m256 sum2 = _mm256_hadd_ps(sum1, sum1);
	const __m128 sum3 = _mm256_extractf128_ps(sum2, 1);
	const __m128 sum4 = _mm_add_ps(_mm256_castps256_ps128(sum2), sum3);
	
	float result;
	_mm_store_ss(&result, sum4);
	return result;
}

static inline double HorizontalAddAVX2_pd(__m256d vector)
{
#if 0
	alignas(32) double buf[4];
	_mm256_storeu_pd(buf, vector);
	return buf[0] + buf[1] + buf[2] + buf[3];
#else
	__m256d sum = _mm256_hadd_pd(vector, vector);
	sum = _mm256_hadd_pd(sum, sum);

	__m128d sum128 = _mm256_extractf128_pd(sum, 0);
	double result = _mm_cvtsd_f64(sum128);
	return result;
#endif
}

template<typename AccumT>
static inline AccumT HorizontalAddAVX2_epi32(__m256i vector)
{
	#if 0
	const __m256i sum1 = _mm256_hadd_epi32(vector, vector);
	const __m256i sum2 = _mm256_hadd_epi32(sum1, sum1);
	const __m128i sum3 = _mm256_extracti128_si256(sum2, 1);
	const __m128i sum4 = _mm_add_epi32(_mm256_castsi256_si128(sum2), sum3);
	
	int32_t result;
	_mm_store_epi32(&result, sum4);
	return result;
	#endif

	alignas(32) int32_t buf[8];
	//_mm256_storeu_epi32(buf, vector);
	_mm256_storeu_si256((__m256i*) buf, vector);

	AccumT sum = 0;
	for(int i = 0; i < 8; i++)
		sum += buf[i];

	return sum;
}

template<typename AccumT>
static inline AccumT HorizontalAddAVX2_epi16(__m256i vector)
{
	alignas(32) int16_t buf[16];
	//_mm256_storeu_epi16(buf, vector);
	_mm256_storeu_si256((__m256i*) buf, vector);

	AccumT sum = 0;
	for(int i = 0; i < 16; i++)
		sum += buf[i];

	return sum;
}

template<typename AccumT>
static inline AccumT HorizontalAddAVX2_epi8(__m256i vector)
{
	alignas(32) int8_t buf[32];
	//_mm256_storeu_epi8(buf, vector);
	_mm256_storeu_si256((__m256i*) buf, vector);

	
	AccumT sum = 0;
	for(int i = 0; i < 32; i++)
		sum += buf[i];

	return sum;
}

static inline int64_t HorizontalAddAVX2_si64(__m256i vector)
{
	alignas(32) int64_t buf[4];
	//_mm256_storeu_epi64(buf, vector);
	_mm256_storeu_si256((__m256i*) buf, vector);
	return buf[0] + buf[1] + buf[2] + buf[3];
}

template<>
template<typename AccumT>
AccumT BlasTensorImpl<float, AVX2>::sum(const float* FUNZEL_RESTRICT data, size_t n, size_t stride)
{
	AccumT accum(0);

	alignas(32) float buffer[8];
	__m256 vectorSum = _mm256_setzero_ps();

	// First: Add vectorized
	size_t off = 0;
	for(; off < n*stride - 7; off += stride*8)
	{
		for(int i = 0; i < 8; i++)
			buffer[i] = data[off + i*stride];

		__m256 values = _mm256_load_ps(buffer);
		vectorSum = _mm256_add_ps(vectorSum, values);
	}

	// Second: Add horizontally
	accum = HorizontalAddAVX2_ps(vectorSum);

	// Third: Accumulate remaining elements
	for(; off < n*stride; off += stride)
		accum += data[off];

	return accum;
}

template<>
template<typename AccumT>
AccumT BlasTensorImpl<double, AVX2>::sum(const double* FUNZEL_RESTRICT data, size_t n, size_t stride)
{
	AccumT accum(0);

	alignas(32) double buffer[4];
	__m256d vectorSum = _mm256_setzero_pd();

	// First: Add vectorized
	size_t off = 0;
	for(; off < n*stride - 3; off += stride*4)
	{
		for(int i = 0; i < 4; i++)
			buffer[i] = data[off + i*stride];

		__m256d values = _mm256_load_pd(buffer);
		vectorSum = _mm256_add_pd(vectorSum, values);
	}

	// Second: Add horizontally
	accum = HorizontalAddAVX2_pd(vectorSum);

	// Third: Accumulate remaining elements
	for(; off < n*stride; off += stride)
		accum += data[off];

	return accum;
}

template<>
template<typename AccumT>
AccumT BlasTensorImpl<int32_t, AVX2>::sum(const int32_t* FUNZEL_RESTRICT data, size_t n, size_t stride)
{
	AccumT accum(0);

	alignas(32) int32_t buffer[8];
	__m256i vectorSum = _mm256_setzero_si256();

	// First: Add vectorized
	size_t off = 0;
	for(; off < n*stride - 7; off += stride*8)
	{
		for(int i = 0; i < 8; i++)
			buffer[i] = data[off + i*stride];

		__m256i values = _mm256_load_si256((__m256i*) buffer); //_mm256_load_epi32(buffer);
		vectorSum = _mm256_add_epi32(vectorSum, values);
	}

	// Second: Add horizontally
	accum = HorizontalAddAVX2_epi32<AccumT>(vectorSum);

	// Third: Accumulate remaining elements
	for(; off < n*stride; off += stride)
		accum += data[off];

	return accum;
}

template<>
template<typename AccumT>
AccumT BlasTensorImpl<int64_t, AVX2>::sum(const int64_t* FUNZEL_RESTRICT data, size_t n, size_t stride)
{
	AccumT accum(0);

	alignas(32) int64_t buffer[4];
	__m256i vectorSum = _mm256_setzero_si256();

	// First: Add vectorized
	size_t off = 0;
	for(; off < n*stride - 3; off += stride*4)
	{
		for(int i = 0; i < 4; i++)
			buffer[i] = data[off + i*stride];

		__m256i values = _mm256_load_si256((__m256i*) buffer); //_mm256_load_epi64(buffer);
		vectorSum = _mm256_add_epi64(vectorSum, values);
	}

	// Second: Add horizontally
	accum = HorizontalAddAVX2_si64(vectorSum);

	// Third: Accumulate remaining elements
	for(; off < n*stride; off += stride)
		accum += data[off];

	return accum;
}

template<>
template<typename AccumT>
AccumT BlasTensorImpl<int16_t, AVX2>::sum(const int16_t* FUNZEL_RESTRICT data, size_t n, size_t stride)
{
	AccumT accum(0);

	alignas(32) int16_t buffer[16];
	__m256i vectorSum = _mm256_setzero_si256();

	// First: Add vectorized
	size_t off = 0;
	for(; off < n*stride - 15; off += stride*16)
	{
		for(int i = 0; i < 16; i++)
			buffer[i] = data[off + i*stride];

		__m256i values = _mm256_load_si256((__m256i*) buffer); //_mm256_loadu_epi16(buffer);
		vectorSum = _mm256_add_epi16(vectorSum, values);
	}

	// Second: Add horizontally
	accum = HorizontalAddAVX2_epi16<AccumT>(vectorSum);

	// Third: Accumulate remaining elements
	for(; off < n*stride; off += stride)
		accum += data[off];

	return accum;
}

template<>
template<typename AccumT>
AccumT BlasTensorImpl<int8_t, AVX2>::sum(const int8_t* FUNZEL_RESTRICT data, size_t n, size_t stride)
{
	AccumT accum(0);

	alignas(32) int8_t buffer[32];
	__m256i vectorSum = _mm256_setzero_si256();

	// First: Add vectorized
	size_t off = 0;
	for(; off < n*stride - 31; off += stride*32)
	{
		for(int i = 0; i < 32; i++)
			buffer[i] = data[off + i*stride];

		__m256i values = _mm256_load_si256((__m256i*) buffer); //_mm256_loadu_epi8(buffer);
		vectorSum = _mm256_add_epi8(vectorSum, values);
	}

	// Second: Add horizontally
	accum = HorizontalAddAVX2_epi8<AccumT>(vectorSum);

	// Third: Accumulate remaining elements
	for(; off < n*stride; off += stride)
		accum += data[off];

	return accum;
}

#endif

#endif
}
}
