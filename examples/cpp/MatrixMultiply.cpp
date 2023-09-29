#include <funzel/Funzel.hpp>
#include <funzel/Tensor.hpp>
#include <funzel/linalg/Linalg.hpp>

#include <iostream>

using namespace funzel;

int main(int argc, char** argv)
{
	funzel::backend::LoadBackend("Blas");

	auto mat1 = Tensor({3, 3}, {
		1.0f, 0.0f, 1.0f,
		2.0f, 4.0f, 0.0f,
		3.0f, 3.0f, 3.0f
	});

	auto mat2 = Tensor({3, 3}, {
		1.0f, 0.0f, 1.0f,
		2.0f, 4.0f, 0.0f,
		3.0f, 3.0f, 3.0f
	});

	auto mat3 = mat1 * mat2;
	std::cout << mat1 << " * " << mat2 << " = " << mat3 << std::endl;

	return 0;
}
