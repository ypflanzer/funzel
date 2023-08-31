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

#include <string>

namespace funzel
{

/**
 * @brief Defines a set of element types which may be used when defining tensors.
 */
enum DTYPE
{
	DINT16 = 0,
	DINT32,
	DINT64,
	DUINT16,
	DUINT32,
	DUINT64,
	DFLOAT32,
	DFLOAT64,
	DINT8,
	DUINT8,
	NONE,
	DTYPE_MAX
};

/**
 * @brief Determines the size of any DTYPE in bytes.
 * 
 * This function is comparable to the builtin 'sizeof'.
 * 
 * @param dtype The DTYPE to determine the size of.
 * @return The size of the type in bytes.
 */
inline constexpr size_t dtypeSizeof(const DTYPE dtype)
{
	switch(dtype)
	{
		case DINT16:
		case DUINT16: return 2;

		case DFLOAT32:
		case DUINT32:
		case DINT32: return 4;

		case DFLOAT64:
		case DUINT64:
		case DINT64: return 8;
	
		case DINT8:
		case DUINT8:
			return 1;

		case NONE:
		default:
			return 0;
	}
}

/**
 * @brief Converts the given DTYPE to a name string.
 * 
 * @param dtype The DTYPE.
 * @return std::string A string containing the name.
 */
inline std::string dtypeToNativeString(const DTYPE dtype)
{
	switch(dtype)
	{
		case DUINT16: return "ushort";
		case DINT16: return "short";

		case DFLOAT32: return "float";
		case DUINT32: return "uint";
		case DINT32: return "int";

		case DFLOAT64: return "double";
		case DUINT64: return "ulong";
		case DINT64: return "long";
	
		case DINT8: return "char";
		case DUINT8: return "uchar";

		default:
		case NONE: return "void";
	}
}

#ifndef SWIG
/**
 * @brief Converts the given type T into a DTYPE.
 * 
 * @tparam T The type to convert.
 * @return DTYPE The DTYPE equivalent to T.
 */
template<typename T>
DTYPE dtype()
{
	if constexpr(std::is_same_v<T, double>) return DFLOAT64;
	else if constexpr(std::is_same_v<T, float>) return DFLOAT32;
	else if constexpr(std::is_same_v<T, int64_t>) return DINT64;
	else if constexpr(std::is_same_v<T, uint64_t>) return DUINT64;
	else if constexpr(std::is_same_v<T, int>) return DINT32;
	else if constexpr(std::is_same_v<T, unsigned int>) return DUINT32;
	else if constexpr(std::is_same_v<T, short>) return DINT16;
	else if constexpr(std::is_same_v<T, unsigned short>) return DUINT16;
	else if constexpr(std::is_same_v<T, char>) return DINT8;
	else if constexpr(std::is_same_v<T, unsigned char>) return DUINT8;
	return NONE;
}

/**
 * @brief Determines the DTYPE of a given object.
 * 
 * @tparam T The type of the given object.
 * @return DTYPE The DTYPE equivalent to T.
 * @see dtype()
 */
template<typename T>
DTYPE dtypeOf(const T&) { return dtype<T>(); }
#endif

}
