
%module nn;

%{

#include "funzel/nn/Module.hpp"
#include "funzel/nn/Conv2D.hpp"
#include "funzel/nn/Linear.hpp"
#include "funzel/nn/NNBackendTensor.hpp"
#include "funzel/nn/Pool2D.hpp"
#include "funzel/nn/ReLU.hpp"
#include "funzel/nn/Sequential.hpp"
#include "funzel/nn/Sigmoid.hpp"

%}

%include "funzel/nn/Module.hpp"
%include "funzel/nn/Conv2D.hpp"
%include "funzel/nn/Linear.hpp"
%include "funzel/nn/NNBackendTensor.hpp"
%include "funzel/nn/Pool2D.hpp"
%include "funzel/nn/ReLU.hpp"
%include "funzel/nn/Sequential.hpp"
%include "funzel/nn/Sigmoid.hpp"
