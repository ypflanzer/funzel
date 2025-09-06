
%module linalg;

%{

#include "funzel/linalg/Linalg.hpp"
#include "funzel/linalg/LinalgBackendTensor.hpp"

%}

%include "funzel/linalg/Linalg.hpp"
%include "funzel/linalg/LinalgBackendTensor.hpp"

#ifdef SWIGLUA

%luacode {
	local origSvd = LuaFunzel.funzel.linalg.svd
	function LuaFunzel.funzel.linalg.svd(matrix)
		-- Do SVD and unpack results
		local svdresult = origSvd(matrix)
		return svdresult.u, svdresult.s, svdresult.vh
	end
}

#endif
