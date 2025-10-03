
%module linalg;

%{

#include "funzel/linalg/Linalg.hpp"
#include "funzel/linalg/LinalgBackendTensor.hpp"

%}

%include "funzel/linalg/Linalg.hpp"
%include "funzel/linalg/LinalgBackendTensor.hpp"

#if 0

%luacode {
	if false then
	local origSvd = LuaFunzel.funzel.linalg.svd
	function LuaFunzel.funzel.linalg.svd(matrix, fullMatrices)
		if fullMatrices == nil then
			fullMatrices = true
		end

		-- Do SVD and unpack results
		local svdresult = origSvd(matrix, fullMatrices)
		-- return LuaFunzel.funzel.Tensor(svdresult.u), LuaFunzel.funzel.Tensor(svdresult.s), LuaFunzel.funzel.Tensor(svdresult.vh)
		--return svdresult.u:clone(), svdresult.s:clone(), svdresult.vh:clone()
	end
	end
}

#endif
