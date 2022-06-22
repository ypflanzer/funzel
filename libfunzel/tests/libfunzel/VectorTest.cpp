#include <gtest/gtest.h>
#include <funzel/Vector.hpp>

using namespace funzel;

TEST(Vector, Init)
{
	Vec3 v{1, 2, 3};
	for(int i = 0; i < v.size(); i++)
		EXPECT_EQ(v[i], i+1);
}
