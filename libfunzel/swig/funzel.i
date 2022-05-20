%{
#include <funzel/Tensor.hpp>

using namespace funzel;
using namespace std;

%}

%include <typemaps.i>
%include <std_vector.i>

%include <typemaps.i>
%include <std_vector.i>
%include <std_string.i>
%include <std_except.i>

%feature("autodoc", "3");
%include "funzel/Tensor.hpp"

%extend funzel::Tensor {
	std::string __str__()
	{
		return $self->toString();
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
