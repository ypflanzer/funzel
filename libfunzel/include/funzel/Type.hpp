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
#include <stdexcept>
#include <cstdint>

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
		case DUINT8: return 1;

		case NONE: return 0;
		default: throw std::invalid_argument("Invalid DTYPE given: " + std::to_string(dtype));
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

		case NONE: return "void";

		default:
			return "Unknown (" + std::to_string(dtype) + ")";
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
	else if constexpr(std::is_same_v<T, int32_t>) return DINT32;
	else if constexpr(std::is_same_v<T, uint32_t>) return DUINT32;
	else if constexpr(std::is_same_v<T, int16_t>) return DINT16;
	else if constexpr(std::is_same_v<T, uint16_t>) return DUINT16;
	else if constexpr(std::is_same_v<T, int8_t>) return DINT8;
	else if constexpr(std::is_same_v<T, uint8_t>) return DUINT8;
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

#ifndef SWIG
template<typename Fn, typename... Args>
inline void DoAsDtype(DTYPE dtype, Fn&& fn, Args&&... args)
{
	// FIXME: This always fails. Find out why!
	//static_assert(std::is_invocable_v<Fn, Args&&...>,
	//				"The given function needs the following signature: void(args...)");
	
	switch(dtype)
	{
	case DINT16: fn.template operator()<int16_t>(args...); break;
	case DINT32: fn.template operator()<int32_t>(args...); break;
	case DINT64: fn.template operator()<int64_t>(args...); break;

	case DUINT16: fn.template operator()<uint16_t>(args...); break;
	case DUINT32: fn.template operator()<uint32_t>(args...); break;
	case DUINT64: fn.template operator()<uint64_t>(args...); break;

	case DFLOAT32: fn.template operator()<float>(args...); break;
	case DFLOAT64: fn.template operator()<double>(args...); break;
	
	case DINT8: fn.template operator()<int8_t>(args...); break;
	case DUINT8: fn.template operator()<uint8_t>(args...); break;

	default:
		throw std::runtime_error("Invalid DTYPE given: " + dtypeToNativeString(dtype));
	}
}
#endif

#endif

}
