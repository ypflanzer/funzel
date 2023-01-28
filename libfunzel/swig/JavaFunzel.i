%module Funzel
%{
#include <funzel/Funzel.hpp>
%}

%include <typemaps.i>
%include <std_string.i>
%include <std_vector.i>
%include <std_array.i>
%include <carrays.i>

%include "arrays_java.i"

// Does not work for now.
// %nspace;

%include "funzel.i"
%include "funzel_nn.i"
%include "funzel_cv.i"
