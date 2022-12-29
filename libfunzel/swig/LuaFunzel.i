%module LuaFunzel
%{
#include <funzel/Funzel.hpp>
#include <iostream>
%}

%include <typemaps.i>
%include <std_string.i>
%include <std_vector.i>
%include <carrays.i>

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
