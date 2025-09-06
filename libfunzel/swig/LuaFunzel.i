%module LuaFunzel
%{
#include <funzel/Funzel.hpp>
#include <iostream>

// For memcpy
#include <cstring>

%}

%include <typemaps.i>
%include <std_string.i>
%include <std_vector.i>
%include <carrays.i>

// Global functions do not work correctly.
// Only use static classes for those?
%nspace;

// Based on: https://github.com/swig/swig/issues/591
// Rename functions in namespaces ("_NSFUNC_" prefix and "::" -> "_")
%rename("%(regex:/^(\\w+)::(\\w+)$/_NSFUNC_\\1_\\2/)s",
		regextarget=1,
		fullname=1,
		%$isfunction,
		%$not %$ismember) "^\\w+::\\w+$";
	%rename("%(regex:/^(\\w+)::(\\w+)::(\\w+)$/_NSFUNC_\\1_\\2_\\3/)s",
		regextarget=1,
		fullname=1,
		%$isfunction,
		%$not %$ismember) "^\\w+::\\w+::\\w+$";

%luacode %{
	local function SplitString(str, separator)
		separator = separator or ":"
		local fields = {}
		local start = 1

		while true do
			local sepStart, sepEnd = string.find(str, separator, start, true)
			if not sepStart then
				table.insert(fields, string.sub(str, start))
				break
			end

			table.insert(fields, string.sub(str, start, sepStart - 1))
			start = sepEnd + 1
		end
		return fields
	end

	-- Move functions in namespaces into tables
	for name, func in pairs(LuaFunzel) do
		if type(func) == "function" and name:sub(1, 8) == "_NSFUNC_" then
			-- Strip "_NSFUNC_" prefix and split name at "_"
			local nameParts = SplitString(name:sub(9, -1), "_")

			-- Table to put function or sub-table into (starts at the module)
			local parent = LuaFunzel

			-- Loop over the name's parts
			for i, namePart in ipairs(nameParts) do
				if i == #nameParts then
					-- Insert function
					parent[namePart] = func
				else
					-- Create table
					parent[namePart] = parent[namePart] or {}
				end

				parent = parent[namePart]
			end

			-- Delete original function
			LuaFunzel[name] = nil
		end
	end
%}

namespace std
{
%template(IntVector) vector<unsigned int>;
%template(FloatVector) vector<float>;
%template(DoubleVector) vector<double>;
%template(SizeVector) vector<size_t>;
}

%include "LuaTables.i"
SWIG_TYPEMAP_NUM_VECTOR(float)
SWIG_TYPEMAP_NUM_VECTOR(double)
SWIG_TYPEMAP_NUM_VECTOR(int)
SWIG_TYPEMAP_NUM_VECTOR(unsigned int)
SWIG_TYPEMAP_NUM_VECTOR(size_t)

//SWIG_TYPEMAP_NUM_CUSTOM(funzel::Shape, size_t)
//SWIG_TYPEMAP_NUM_CUSTOM(funzel::Shape*, size_t)
SWIG_TYPEMAP_NUM_CUSTOM(funzel::Shape&, size_t)
//SWIG_TYPEMAP_NUM_CUSTOM(const funzel::Shape&, size_t)

%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) funzel::Shape&
{
	$1 = lua_istable(L, $input);
}

%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) funzel::Shape
{
	$1 = lua_istable(L, $input);
}

%include "funzel.i"
%include "funzel_nn.i"
%include "funzel_cv.i"
%include "funzel_linalg.i"

%extend funzel::Tensor
{
	funzel::Tensor __add__(const funzel::Tensor& other)
	{
		return *$self + other;
	}
	
	funzel::Tensor __sub__(const funzel::Tensor& other)
	{
		return *$self - other;
	}
	
	funzel::Tensor __mul__(const funzel::Tensor& other)
	{
		return *$self * other;
	}
	
	funzel::Tensor __div__(const funzel::Tensor& other)
	{
		return *$self / other;
	}

	funzel::Tensor __div__(double other)
	{
		return *$self / other;
	}

	funzel::Tensor __neg__()
	{
		return -(*$self);
	}

	/*
	funzel::Tensor __pow__(double exponent)
	{
		return $self->pow(exponent);
	}
	*/

	funzel::Tensor __concat__(const funzel::Tensor& other)
	{
		return $self->matmul(other);
	}
}

#if 0
%inline %{
#include <funzel/Tensor.hpp>
#include <cstring>

namespace funzel
{

static funzel::Tensor frombufferWrapped(const std::string& data, funzel::DTYPE dtype)
{
	const size_t typeSize = funzel::dtypeSizeof(dtype);
	if (typeSize == 0)
	{
		throw std::runtime_error("Unknown dtype!");
	}

	if (data.size() % typeSize != 0)
	{
		throw std::runtime_error("Data size is not a multiple of dtype size!");
	}

	const size_t numElements = data.size() / typeSize;

	funzel::Tensor tensor = funzel::Tensor::empty({numElements}, dtype);
	std::memcpy(tensor.data(), data.data(), data.size());
	return tensor;
}

}
%}
#endif

// Load a Lua string into a Tensor without copying multiple times
// as is the case for the wrapped version with std::string.
%native(_NSFUNC_funzel_frombuffer) int frombuffer(lua_State* L);
%{
int frombuffer(lua_State* L)
{
	funzel::DTYPE dtype = static_cast<funzel::DTYPE>(lua_tointeger(L, 2));
	size_t dataSize = 0;
	const char* data = lua_tolstring(L, 1, &dataSize);

	if (!data)
	{
		luaL_error(L, "First argument must be a string (buffer)");
		return 0;
	}
	
	const size_t typeSize = funzel::dtypeSizeof(dtype);
	if (typeSize == 0)
	{
		luaL_error(L, "Unknown dtype!");
		return 0;
	}

	if (dataSize % typeSize != 0)
	{
		luaL_error(L, "Data size is not a multiple of dtype size!");
		return 0;
	}

	const size_t numElements = dataSize / typeSize;

	try
	{
		funzel::Tensor* tensor = new funzel::Tensor(funzel::Tensor::empty({numElements}, dtype));

		std::memcpy(tensor->data(), data, dataSize);

		SWIG_NewPointerObj(L, tensor, SWIG_TypeQuery("funzel::Tensor *"), 1);
		return 1;
	}
	catch (const std::exception& e)
	{
		luaL_error(L, e.what());
		return 0;
	}

	return 1;
}
%}

