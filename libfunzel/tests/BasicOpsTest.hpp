
#define FUNZEL_TEST_(prefix, suite, name) TYPED_TEST(prefix, suite##_##name)
#define FUNZEL_TEST(prefix, suite, name) FUNZEL_TEST_(prefix, suite, name)

using vector_typelist = testing::Types<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double>;
template<class> struct CommonTest : testing::Test {};
TYPED_TEST_SUITE(CommonTest, vector_typelist);

// **************************************************
// INSTANTIATION
// **************************************************
FUNZEL_TEST(CommonTest, BasicOps, Instantiation)
{
	auto v = Tensor::empty({3, 3, 3}, funzel::dtype<TypeParam>()).to(TestDevice);
	EXPECT_EQ(v.shape, (Shape{3,3,3}));

	const auto st = sizeof(TypeParam);
	EXPECT_EQ(v.strides, (Shape{3*3*st, 3*st, st}));
}

// **************************************************
// TYPE CONVERSION
// **************************************************
FUNZEL_TEST(CommonTest, BasicOps, Astype)
{
	Tensor v = Tensor({27},
	{
		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,

		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,
		
		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,
	}).to(TestDevice).astype<TypeParam>();

	auto vcpu = v.cpu();
	for(int i = 0; i < 27; i++)
	{
		EXPECT_EQ(v[i].item<int>(), 2);
	}
}

// **************************************************
// ADDITION
// **************************************************
FUNZEL_TEST(CommonTest, BasicOps, AddMatrix)
{
	auto v = Tensor::ones({3, 3, 3}, funzel::dtype<TypeParam>()).to(TestDevice);
	auto result = v.add(v);

	v = v.cpu();
	result = result.cpu();

	const Tensor expected = Tensor({3, 3, 3},
	{
		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,

		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,
		
		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,
	}).astype<TypeParam>();

	// Make sure the original value did not change
	const auto ones = Tensor::ones({3, 3, 3}, funzel::dtype<TypeParam>());
	EXPECT_TENSOR_EQ(v, ones);
	EXPECT_TENSOR_EQ(result, expected);
}

FUNZEL_TEST(CommonTest, BasicOps, AddMatrixInplace)
{
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice);
	v.add_(v);
	v = v.cpu();

	const Tensor expected({3, 3, 3},
	{
		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,

		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,
		
		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,
	});

	EXPECT_TENSOR_EQ(v, expected);
}

FUNZEL_TEST(CommonTest, BasicOps, AddMatrixStridedInplace)
{
	const auto ones = Tensor::ones({3, 3}).to(TestDevice);
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice).transpose();
	v[1].add_(ones);
	v = v.transpose();
	v = v.cpu();

	for(size_t p = 0; p < 3; p++)
		for(size_t q = 0; q < 3; q++)
			for(size_t r = 0; r < 3; r++)
			{
				EXPECT_EQ((v[{p, q, r}].item<float>()), (r == 1 ? 2 : 1));
			}
}

// **************************************************
// SUBTRACTION
// **************************************************
FUNZEL_TEST(CommonTest, BasicOps, SubtractMatrix)
{
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice);
	auto result = v.sub(v);

	v = v.cpu();
	result = result.cpu();

	const Tensor expected({3, 3, 3},
	{
		0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f,

		0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f,

		0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f,
	});

	// Make sure the original value did not change
	EXPECT_TENSOR_EQ(v, Tensor::ones({3, 3, 3}));
	EXPECT_TENSOR_EQ(result, expected);
}

FUNZEL_TEST(CommonTest, BasicOps, SubtractMatrixInplace)
{
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice);
	v.sub_(v);
	v = v.cpu();

	const Tensor expected({3, 3, 3},
	{
		0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f,

		0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f,

		0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f,
	});

	EXPECT_TENSOR_EQ(v, expected);
}

FUNZEL_TEST(CommonTest, BasicOps, SubtractMatrixStridedInplace)
{
	const auto ones = Tensor::ones({3, 3}).to(TestDevice);
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice).transpose();
	v[1].sub_(ones);
	v = v.transpose();
	v = v.cpu();

	for(size_t p = 0; p < 3; p++)
		for(size_t q = 0; q < 3; q++)
			for(size_t r = 0; r < 3; r++)
			{
				EXPECT_EQ((v[{p, q, r}].item<float>()), (r == 1 ? 0 : 1));
			}
}

// **************************************************
// MULTIPLICATION
// **************************************************
FUNZEL_TEST(CommonTest, BasicOps, MultiplyMatrix)
{
	auto v = Tensor::ones({3, 3, 3}).to(TestDevice);
	auto result = v.sub(v);

	v = v.cpu();
	result = result.cpu();

	const Tensor expected({3, 3, 3},
	{
		0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f,

		0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f,

		0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f,
	});

	// Make sure the original value did not change
	EXPECT_TENSOR_EQ(v, Tensor::ones({3, 3, 3}));
	EXPECT_TENSOR_EQ(result, expected);
}

FUNZEL_TEST(CommonTest, BasicOps, MultiplyMatrixInplace)
{
	auto v = Tensor({3, 3, 3},
	{
		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,

		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,
		
		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,
	}).to(TestDevice);

	v.mul_(v);
	v = v.cpu();

	const Tensor expected({3, 3, 3},
	{
		9.0f, 9.0f, 9.0f,
		9.0f, 9.0f, 9.0f,
		9.0f, 9.0f, 9.0f,

		9.0f, 9.0f, 9.0f,
		9.0f, 9.0f, 9.0f,
		9.0f, 9.0f, 9.0f,

		9.0f, 9.0f, 9.0f,
		9.0f, 9.0f, 9.0f,
		9.0f, 9.0f, 9.0f,
	});

	EXPECT_TENSOR_EQ(v, expected);
}

FUNZEL_TEST(CommonTest, BasicOps, MultiplyMatrixStridedInplace)
{
	const auto three = Tensor({3, 3},
	{
		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,
	}).to(TestDevice);

	auto v = Tensor({3, 3, 3},
	{
		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,

		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,
		
		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,
	}).to(TestDevice).transpose();

	v[1].mul_(three);
	v = v.transpose();
	v = v.cpu();

	for(size_t p = 0; p < 3; p++)
		for(size_t q = 0; q < 3; q++)
			for(size_t r = 0; r < 3; r++)
			{
				EXPECT_EQ((v[{p, q, r}].item<float>()), (r == 1 ? 9 : 3));
			}
}

// **************************************************
// DIVISION
// **************************************************
FUNZEL_TEST(CommonTest, BasicOps, DivideMatrix)
{
	auto v = Tensor({3, 3, 3},
	{
		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,

		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,
		
		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,
	});

	auto v1 = v.to(TestDevice);
	auto v2 = Tensor({3, 3, 3},
	{
		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,

		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,
		
		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,
	}).to(TestDevice);

	auto result = v1.div(v2);

	v1 = v1.cpu();
	result = result.cpu();

	const Tensor expected({3, 3, 3},
	{
		1.5f, 1.5f, 1.5f,
		1.5f, 1.5f, 1.5f,
		1.5f, 1.5f, 1.5f,

		1.5f, 1.5f, 1.5f,
		1.5f, 1.5f, 1.5f,
		1.5f, 1.5f, 1.5f,

		1.5f, 1.5f, 1.5f,
		1.5f, 1.5f, 1.5f,
		1.5f, 1.5f, 1.5f,
	});

	// Make sure the original value did not change
	EXPECT_TENSOR_EQ(v1, v);
	EXPECT_TENSOR_EQ(result, expected);
}

FUNZEL_TEST(CommonTest, BasicOps, DivideMatrixInplace)
{
	auto v1 = Tensor({3, 3, 3},
	{
		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,

		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,
		
		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,
	}).to(TestDevice);

	auto v2 = Tensor({3, 3, 3},
	{
		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,

		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,
		
		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,
	}).to(TestDevice);

	v1.div_(v2);
	v1 = v1.cpu();

	const Tensor expected({3, 3, 3},
	{
		1.5f, 1.5f, 1.5f,
		1.5f, 1.5f, 1.5f,
		1.5f, 1.5f, 1.5f,

		1.5f, 1.5f, 1.5f,
		1.5f, 1.5f, 1.5f,
		1.5f, 1.5f, 1.5f,

		1.5f, 1.5f, 1.5f,
		1.5f, 1.5f, 1.5f,
		1.5f, 1.5f, 1.5f,
	});

	EXPECT_TENSOR_EQ(v1, expected);
}

FUNZEL_TEST(CommonTest, BasicOps, DivideMatrixStridedInplace)
{
	const auto three = Tensor({3, 3},
	{
		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,
		2.0f, 2.0f, 2.0f,
	}).to(TestDevice);

	auto v = Tensor({3, 3, 3},
	{
		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,

		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,
		
		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,
		3.0f, 3.0f, 3.0f,
	}).to(TestDevice).transpose();

	v[1].div_(three);
	v = v.transpose();
	v = v.cpu();

	for(size_t p = 0; p < 3; p++)
		for(size_t q = 0; q < 3; q++)
			for(size_t r = 0; r < 3; r++)
			{
				EXPECT_FLOAT_EQ((v[{p, q, r}].item<float>()), (r == 1 ? 1.5f : 3));
			}
}
