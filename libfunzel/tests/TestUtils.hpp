#pragma once

#define EXPECT_TENSOR_EQ(t1, t2) \
ASSERT_EQ((t1).shape, (t2).shape); \
ASSERT_EQ((t1).dtype, (t2).dtype); \
	for(size_t i = 0; i < (t1).size(); i++) \
	{ \
		float* cdata = (float*) (t1).data(i*sizeof(float)); \
		float* edata = (float*) (t2).data(i*sizeof(float)); \
		EXPECT_EQ(*cdata, *edata); \
	}
