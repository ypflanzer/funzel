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

%typemap(in) funzel::small_vector<funzel::TensorSlice> ()
{
	const int stackpos = $input;

	size_t n = lua_rawlen(L, stackpos);
	$1.reserve(n);

	for (size_t i = 1; i <= n; ++i)
	{
		lua_rawgeti(L, stackpos, i);
		if (!lua_istable(L, -1))
		{
			luaL_error(L, "Each slice must be a table");
			lua_pop(L, 1);
			return 0;
		}

		const auto len = lua_rawlen(L, -1);
		funzel::TensorSlice slice;

		if (len == 0)
		{
			slice = funzel::TensorSlice();
		}
		else if (len == 1)
		{
			lua_rawgeti(L, -1, 1);
			int start = lua_tointeger(L, -1);
			lua_pop(L, 1);
			slice = funzel::TensorSlice(start, start + 1, 1);
		}
		else if (len == 2)
		{
			lua_rawgeti(L, -1, 1);
			int start = lua_tointeger(L, -1);
			lua_pop(L, 1);
			lua_rawgeti(L, -1, 2);
			int stop = lua_tointeger(L, -1);
			lua_pop(L, 1);
			slice = funzel::TensorSlice(start, stop, 1);
		}
		else if (len == 3)
		{
			lua_rawgeti(L, -1, 1);
			int start = lua_tointeger(L, -1);
			lua_pop(L, 1);
			lua_rawgeti(L, -1, 2);
			int stop = lua_tointeger(L, -1);
			lua_pop(L, 1);
			lua_rawgeti(L, -1, 3);
			int step = lua_tointeger(L, -1);
			lua_pop(L, 1);
			slice = funzel::TensorSlice(start, stop, step);
		}
		else
		{
			luaL_error(L, "Slice table must have 0, 1, 2, or 3 elements");
			lua_pop(L, 1);
			return 0;
		}

		$1.push_back(slice);
		lua_pop(L, 1);
	}
}

%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) const funzel::small_vector<funzel::TensorSlice>&
{
	$1 = lua_istable(L, $input);
}

%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) const funzel::small_vector<funzel::TensorSlice>
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
		return $self->mul(other);
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

	funzel::Tensor __call__(const funzel::small_vector<funzel::TensorSlice> slices)
	{
		return $self->slice(slices);
	}
}

#if 1
%inline %{
#include <funzel/Tensor.hpp>
#include <cstring>

namespace funzel
{

static funzel::Tensor frombuffer(const std::string& data, funzel::DTYPE dtype)
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
#else
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
#endif

#ifdef SWIGLUA

%luacode {

	-- Extend the Tensor class with Lua methods
	local mt = getmetatable(LuaFunzel.funzel.Tensor)
	local orig_index = mt.__index

	mt.__index = function(self, key)

		print("__index: ", key)
		-- Check if there is a Lua extension method
		local ext = rawget(mt, "__lua_ext")
		print("ext: ", ext[key])
		if ext and ext[key] then
			print("Found extension method: ", key)
			return ext[key]
		end

		-- Fallback to original __index
		local val = orig_index(self, key)
		print("val: ", val)

		if val ~= nil then
			return val
		end

		return nil
	end

	mt.__call = function(self, ...)
		print("__call")
		return self:slice(...)
	end

	-- Table to hold Lua-side extensions
	mt.__lua_ext = {}

	setmetatable(LuaFunzel.funzel.Tensor, mt)
}

#endif

%{
	namespace TensorExtensions
	{
		int slice(lua_State* L)
		{
			funzel::Tensor* me = nullptr;
			if(!SWIG_isptrtype(L,1))
				return 0;

			if (!SWIG_IsOK(SWIG_ConvertPtr(L, 1, (void**) &me, SWIGTYPE_p_funzel__Tensor, 0)))
			{
				return 0;
			}

			if (!me)
			{
				luaL_error(L, "Invalid Tensor object");
				return 0;
			}

			funzel::small_vector<funzel::TensorSlice> slices;
			if (lua_istable(L, 2))
			{
				int n = lua_rawlen(L, 2);
				slices.reserve(n);
				
				for (int i = 1; i <= n; ++i)
				{
					lua_rawgeti(L, 2, i);
					if (!lua_istable(L, -1))
					{
						luaL_error(L, "Each slice must be a table");
						return 0;
					}

					int len = lua_rawlen(L, -1);
					funzel::TensorSlice slice;

					if (len == 0)
					{
						// ":" full slice
						slice = funzel::TensorSlice();
					}
					else if (len == 1)
					{
						lua_rawgeti(L, -1, 1);
						int start = lua_tointeger(L, -1);
						lua_pop(L, 1);
						slice = funzel::TensorSlice(start, start + 1, 1);
					}
					else if (len == 2)
					{
						lua_rawgeti(L, -1, 1);
						int start = lua_tointeger(L, -1);
						lua_pop(L, 1);
						lua_rawgeti(L, -1, 2);
						int stop = lua_tointeger(L, -1);
						lua_pop(L, 1);
						slice = funzel::TensorSlice(start, stop, 1);
					}
					else if (len == 3)
					{
						lua_rawgeti(L, -1, 1);
						int start = lua_tointeger(L, -1);
						lua_pop(L, 1);
						lua_rawgeti(L, -1, 2);
						int stop = lua_tointeger(L, -1);
						lua_pop(L, 1);
						lua_rawgeti(L, -1, 3);
						int step = lua_tointeger(L, -1);
						lua_pop(L, 1);
						slice = funzel::TensorSlice(start, stop, step);
					}
					else
					{
						luaL_error(L, "Slice table must have 0, 1, 2, or 3 elements");
						return 0;
					}

					slices.push_back(slice);
					lua_pop(L, 1);
				}
			}
			else
			{
				luaL_error(L, "Second argument must be a table of slices");
				return 0;
			}

			funzel::Tensor* result = new funzel::Tensor(me->slice(slices));
			SWIG_NewPointerObj(L, result, SWIG_TypeQuery("funzel::Tensor *"), 1);
			return 1;
		}
	}
%}

%wrapper %{
// https://stackoverflow.com/questions/16360012/swig-lua-extending-extend-class-with-native-is-it-possible-to-add-native
void script_addNativeMethod(lua_State *L, const char *className, const char *methodName, lua_CFunction fn)
{
	SWIG_Lua_get_class_registry(L); /* get the registry */
	lua_pushstring(L, className);   /* get the name */
	lua_rawget(L,-2);               /* get the metatable itself */
	lua_remove(L,-2);               /* tidy up (remove registry) */

	// If the metatable is not null, add the method to the ".fn" table
	if(lua_isnil(L, -1) != 1)
	{
		SWIG_Lua_get_table(L, ".fn");
		SWIG_Lua_add_function(L, methodName, fn);
		lua_pop(L, 2);              /* tidy up (remove metatable and ".fn" table) */
	}
	else
	{
		printf("[script_addNativeMethod(..)] - \"%s\" metatable is not found. Method \"%s\" will not be added\n", className, methodName);
		return;
	}
}

void script_addNativeMethod2(lua_State *L, const char *className, const char *methodName, lua_CFunction fn)
{
	SWIG_Lua_get_class_registry(L); /* get the registry */
	lua_pushstring(L, className);   /* get the name */
	lua_rawget(L,-2);               /* get the metatable itself */
	lua_remove(L,-2);               /* tidy up (remove registry) */

	// If the metatable is not null, add the method to the ".fn" table
	if(lua_isnil(L, -1) != 1)
	{
		SWIG_Lua_get_table(L, ".fn");
		SWIG_Lua_add_function(L, methodName, fn);
		lua_pop(L, 2);              /* tidy up (remove metatable and ".fn" table) */
	}
	else
	{
		printf("[script_addNativeMethod(..)] - \"%s\" metatable is not found. Method \"%s\" will not be added\n", className, methodName);
		return;
	}
}
%}

%init %{
	//script_addNativeMethod(L, "funzel.Tensor", "slice", TensorExtensions::slice);
%}
