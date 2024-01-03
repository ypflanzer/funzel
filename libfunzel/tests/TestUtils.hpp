#pragma once

#define EXPECT_TENSOR_EQ(t1, t2) \
ASSERT_EQ((t1).shape, (t2).shape); \
ASSERT_EQ((t1).dtype, (t2).dtype); \
	{ \
		auto t1cont = (t1).unravel(); \
		auto t2cont = (t2).unravel(); \
		for(size_t i = 0; i < (t1).size(); i++) \
		{ \
			float* cdata = (float*) (t1cont).data(i*sizeof(float)); \
			float* edata = (float*) (t2cont).data(i*sizeof(float)); \
			EXPECT_EQ(*cdata, *edata); \
		}\
	}
