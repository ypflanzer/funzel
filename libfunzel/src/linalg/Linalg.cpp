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
#include <funzel/linalg/Linalg.hpp>
#include <funzel/linalg/LinalgBackendTensor.hpp>
#include <funzel/Tensor.hpp>
#include <numeric>

using namespace funzel;
using namespace linalg;

Tensor& funzel::linalg::det(Tensor input, Tensor& result)
{
	input.ensureBackend<LinalgBackendTensor>().det(input, result);
	return result;
}

Tensor funzel::linalg::det(Tensor input)
{
	const auto outsize = std::accumulate(input.shape.begin(), input.shape.end()-2, size_t(1), [](auto a, auto b) { return a*b; });
	auto tgt = Tensor::empty({outsize}, nullptr, input.dtype, input.device);
	input.ensureBackend<LinalgBackendTensor>().det(input, tgt);
	return tgt;
}

Tensor& funzel::linalg::inv(Tensor input, Tensor& result)
{
	input.ensureBackend<LinalgBackendTensor>().inv(input, result);
	return result;
}

Tensor funzel::linalg::inv(Tensor input)
{
	auto tgt = Tensor::empty_like(input);
	input.ensureBackend<LinalgBackendTensor>().inv(input, tgt);
	return tgt;
}

Tensor& funzel::linalg::trace(Tensor input, Tensor& result)
{
	input.ensureBackend<LinalgBackendTensor>().trace(input, result);
	return result;
}

Tensor funzel::linalg::trace(Tensor input)
{
	const auto outsize = std::accumulate(input.shape.begin(), input.shape.end()-2, size_t(1), [](auto a, auto b) { return a*b; });
	auto tgt = Tensor::empty({outsize}, input.dtype, input.device);

	input.ensureBackend<LinalgBackendTensor>().trace(input, tgt);
	return tgt;
}

void funzel::linalg::svd(Tensor input, Tensor& U, Tensor& S, Tensor& V)
{
	input.ensureBackend<LinalgBackendTensor>().svd(input, U, S, V);
}

std::tuple<Tensor, Tensor, Tensor> funzel::linalg::svd(Tensor input)
{
	std::tuple<Tensor, Tensor, Tensor> usv;
	auto& [u, s, v] = usv;

	const size_t m = input.shape[input.shape.size() - 2];
	const size_t n = input.shape[input.shape.size() - 1];

	const auto outsize = std::accumulate(input.shape.begin(), input.shape.end()-2, size_t(1), [](auto a, auto b) { return a*b; });

	if(outsize != 1)
	{
		s = Tensor::empty({outsize, std::min(m, n)}, input.dtype, input.device);
		u = Tensor::empty({outsize, m, m}, input.dtype, input.device);
		v = Tensor::empty({outsize, n, n}, input.dtype, input.device);
	}
	else
	{
		s = Tensor::empty({std::min(m, n)}, input.dtype, input.device);
		u = Tensor::empty({m, m}, input.dtype, input.device);
		v = Tensor::empty({n, n}, input.dtype, input.device);
	}

	svd(input, u, s, v);

	// Get last indices so the last entries can be transposed.
	// s and v are created as their transpose in col major mode by default.
	const size_t shapeSize = u.shape.size();
	std::swap(u.shape[shapeSize - 2], u.shape[shapeSize - 1]);
	std::swap(v.shape[shapeSize - 2], v.shape[shapeSize - 1]);

	std::swap(u.strides[shapeSize - 2], u.strides[shapeSize - 1]);
	std::swap(v.strides[shapeSize - 2], v.strides[shapeSize - 1]);

	return usv;
}

Tensor funzel::linalg::svdvals(Tensor input)
{
	const auto ndim = input.shape.size();
	const auto ndiag = std::min(input.shape[ndim-1], input.shape[ndim-2]);

	const auto outsize = std::accumulate(input.shape.begin(), input.shape.end()-2, size_t(1), [](auto a, auto b) { return a*b; });
	Tensor s = Tensor::empty({outsize, ndim});
	
	return s;
}
