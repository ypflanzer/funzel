#include <funzel/Funzel.hpp>
#include <funzel/Tensor.hpp>
#include <funzel/linalg/Linalg.hpp>

#include <iostream>

using namespace funzel;

int main(int argc, char** argv)
{
	funzel::backend::LoadBackend("Blas");

	auto A = Tensor({3, 3}, {
		4.0f, 3.0f, 2.0f,
		-2.0f, 2.0f, 3.0f,
		3.0f, -5.0f, 2.0f
	});

	auto B = Tensor({3, 1}, {
		25.0f, -10.0f, -4.0f
	});

	auto X = linalg::inv(A) * B;
	
	std::cout << "The solution to\n"
				"\t| 4x + 3y + 2z = 25\t|\n"
				"\t| -2x + 2y + 3z = -10\t|\n"
				"\t| 3x -5y + 2z = -4\t|\n"
				"is\n\tx=" 
			  << X[0].item<float>() << " y=" << X[1].item<float>() << " z=" << X[2].item<float>() << std::endl;

	return 0;
}
