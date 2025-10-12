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
#pragma once

#include <initializer_list>

namespace funzel
{

template<typename T, unsigned int Num>
struct Vec
{
	Vec() = default;
	
#ifndef SWIG
	template<typename... Args>
	Vec(Args... args): data{static_cast<T>(args)...} {}
#endif

	T& operator[](unsigned int idx) { return data[idx]; }
	const T& operator[](unsigned int idx) const { return data[idx]; }
	unsigned int size() const { return Num; }

	T data[Num];
};

typedef Vec<float, 2> Vec2;
typedef Vec<float, 3> Vec3;
typedef Vec<float, 4> Vec4;

typedef Vec<double, 2> DVec2;
typedef Vec<double, 3> DVec3;
typedef Vec<double, 4> DVec4;

typedef Vec<int, 2> IVec2;
typedef Vec<int, 3> IVec3;
typedef Vec<int, 4> IVec4;

typedef Vec<unsigned int, 2> UVec2;
typedef Vec<unsigned int, 3> UVec3;
typedef Vec<unsigned int, 4> UVec4;

}
