#pragma once

#ifndef EXPECT_TENSOR_EQ
#define EXPECT_TENSOR_EQ(t1, t2) \
ASSERT_EQ((t1).shape, (t2).shape); \
ASSERT_EQ((t1).dtype, (t2).dtype); \
	{ \
		auto t1cont = (t1).unravel().flatten(); \
		auto t2cont = (t2).unravel().flatten(); \
		for(size_t i = 0; i < (t1).size(); i++) \
		{ \
			double cdata = (t1cont)[i].template item<double>(); \
			double edata = (t2cont)[i].template item<double>(); \
			EXPECT_FLOAT_EQ(cdata, edata); \
		}\
	}
#endif
