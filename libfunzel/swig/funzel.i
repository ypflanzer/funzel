%{
#include <funzel/Tensor.hpp>

using namespace funzel;
using namespace std;

%}

%include <typemaps.i>
%include <std_vector.i>
%include <std_string.i>
%include <std_except.i>

%feature("autodoc", "3");

%include "funzel/Funzel.hpp"

%rename(copy) *::operator=;
%rename(add) *::operator+;
%rename(sub) *::operator-;
%rename(mul) *::operator*;
%rename(div) *::operator/;
%ignore *::operator[];

%define FUNZEL_API %enddef
%include "funzel/Tensor.hpp"

%extend funzel::Tensor {
	std::string __str__()
	{
		return $self->toString();
	}

	funzel::Tensor __getitem__(unsigned int i)
	{
		return (*$self)[i];
	}

	funzel::Tensor __getitem__(const funzel::Shape& i)
	{
		return (*$self)[i];
	}
}

%extend std::vector<size_t> {
	std::string __str__()
	{
		std::string str = "(";
		for(auto& e : *$self)
			str += std::to_string(e) + (&e != &$self->back() ? ", " : "");

		str += ")";
		return str;
	}
}

%include "funzel/small_vector"
namespace funzel
{
%template(IntSmallVector) small_vector<int>;
%template(FloatSmallVector) small_vector<float>;
%template(DoubleSmallVector) small_vector<double>;
%template(SizeSmallVector) small_vector<size_t>;
}

%extend funzel::small_vector<size_t> {
	std::string __str__()
	{
		std::string str = "(";
		for(auto& e : *$self)
			str += std::to_string(e) + (&e != &$self->back() ? ", " : "");

		str += ")";
		return str;
	}
}
