%module Funzel
%{
#include <funzel/Funzel.hpp>
%}

%include <typemaps.i>
%include <std_string.i>
%include <std_vector.i>
%include <std_array.i>
%include <carrays.i>

%include "arrays_csharp.i"

%define SWIG_TYPEMAP_SMALL_VECTOR(NAME, T)
%typemap(cscode) funzel::small_vector<T> %{
public NAME(T[] values) : this(FunzelPINVOKE.new_##NAME##__SWIG_0(), true) {
	reserve((uint) values.Length);
	foreach(T v in values)
	{
		push_back(v);
	}
}
%}
%enddef

SWIG_TYPEMAP_SMALL_VECTOR(IntSmallVector, int)
SWIG_TYPEMAP_SMALL_VECTOR(FloatSmallVector, float)
SWIG_TYPEMAP_SMALL_VECTOR(DoubleSmallVector, double)

%typemap(cscode) funzel::small_vector<size_t> %{
  public SizeSmallVector(uint[] values) : this(FunzelPINVOKE.new_SizeSmallVector__SWIG_0(), true) {
    reserve((uint) values.Length);
    foreach(uint v in values)
    {
      push_back(v);
    }
  }
%}

%typemap(csclassmodifiers) funzel::Tensor "public partial class"
// %csmethodmodifiers funzel::Tensor::toString "public override";
// %rename funzel::Tensor::toString ToString;

%include "funzel.i"
