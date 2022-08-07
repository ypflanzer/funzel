#include <gtest/gtest.h>
#include <funzel/small_vector>

using namespace funzel;

TEST(small_vector, DefaultConstruct)
{
	small_vector<unsigned int, 5> sv;

	EXPECT_EQ(sv.max_size(), 5);
	EXPECT_EQ(sv.size(), 0);
	EXPECT_TRUE(sv.empty());
}

TEST(small_vector, SizedConstructSmall)
{
	small_vector<unsigned int, 5> sv(3);

	EXPECT_EQ(sv.max_size(), 5);
	EXPECT_EQ(sv.size(), 3);
	EXPECT_FALSE(sv.empty());
}

TEST(small_vector, SizedConstructBig)
{
	small_vector<unsigned int, 5> sv(6);

	EXPECT_GT(sv.max_size(), 5);
	EXPECT_EQ(sv.size(), 6);
	EXPECT_FALSE(sv.empty());
}

TEST(small_vector, InitializerConstruct)
{
	small_vector<unsigned int, 5> sv{1, 2, 3, 4, 5};

	EXPECT_EQ(sv.max_size(), 5);
	EXPECT_FALSE(sv.empty());
	ASSERT_EQ(sv.size(), 5);

	for(int i = 0; i < 5; i++)
	{
		EXPECT_EQ(sv[i], i+1);
	}
}

TEST(small_vector, InitializerConstructBig)
{
	small_vector<unsigned int, 5> sv{1, 2, 3, 4, 5, 6};

	EXPECT_GT(sv.max_size(), 5);
	EXPECT_FALSE(sv.empty());
	ASSERT_EQ(sv.size(), 6);

	for(int i = 0; i < 5; i++)
	{
		EXPECT_EQ(sv[i], i+1);
	}
}

TEST(small_vector, Resize)
{
	small_vector<unsigned int, 5> sv;
	sv.resize(5);

	EXPECT_EQ(sv.max_size(), 5);
	EXPECT_FALSE(sv.empty());
	ASSERT_EQ(sv.size(), 5);
}

TEST(small_vector, ResizeBig)
{
	small_vector<unsigned int, 5> sv;
	sv.resize(6);

	EXPECT_GT(sv.max_size(), 5);
	EXPECT_FALSE(sv.empty());
	ASSERT_EQ(sv.size(), 6);
}

TEST(small_vector, ResizeInit)
{
	small_vector<unsigned int, 5> sv;
	sv.resize(5, 0xDEAD);

	EXPECT_EQ(sv.max_size(), 5);
	EXPECT_FALSE(sv.empty());
	ASSERT_EQ(sv.size(), 5);

	for(int i = 0; i < sv.size(); i++)
		EXPECT_EQ(sv[i], 0xDEAD);
}

TEST(small_vector, ResizeBigInit)
{
	small_vector<unsigned int, 5> sv;
	sv.resize(6, 0xDEAD);

	EXPECT_GT(sv.max_size(), 5);
	EXPECT_FALSE(sv.empty());
	ASSERT_EQ(sv.size(), 6);

	for(int i = 0; i < sv.size(); i++)
		EXPECT_EQ(sv[i], 0xDEAD);
}

TEST(small_vector, Reserve)
{
	small_vector<unsigned int, 5> sv;
	sv.reserve(3);

	EXPECT_EQ(sv.max_size(), 5);
	EXPECT_TRUE(sv.empty());
	ASSERT_EQ(sv.size(), 0);
}

TEST(small_vector, ReserveBig)
{
	small_vector<unsigned int, 5> sv;
	sv.reserve(6);

	EXPECT_EQ(sv.max_size(), 6);
	EXPECT_TRUE(sv.empty());
	ASSERT_EQ(sv.size(), 0);
}

TEST(small_vector, PushBack)
{
	small_vector<unsigned int, 5> sv;

	sv.push_back(1);
	sv.push_back(2);
	sv.push_back(3);

	EXPECT_EQ(sv.max_size(), 5);
	EXPECT_FALSE(sv.empty());
	ASSERT_EQ(sv.size(), 3);
}

TEST(small_vector, PushBackBig)
{
	small_vector<unsigned int, 5> sv;

	for(int i = 1; i <= 6; i++)
		sv.push_back(i);

	EXPECT_GT(sv.max_size(), 5);
	EXPECT_FALSE(sv.empty());
	ASSERT_EQ(sv.size(), 6);
}

TEST(small_vector, EmplaceBack)
{
	small_vector<unsigned int, 5> sv;

	for(int i = 1; i <= 5; i++)
		sv.emplace_back(i);

	EXPECT_EQ(sv.max_size(), 5);
	EXPECT_FALSE(sv.empty());
	ASSERT_EQ(sv.size(), 5);
}

TEST(small_vector, EmplaceBackBig)
{
	small_vector<unsigned int, 5> sv;

	for(int i = 1; i <= 6; i++)
		sv.emplace_back(i);

	EXPECT_GT(sv.max_size(), 5);
	EXPECT_FALSE(sv.empty());
	ASSERT_EQ(sv.size(), 6);
}

TEST(small_vector, Insert)
{
	small_vector<unsigned int, 5> sv(4);

	sv.insert(sv.begin() + 2, 5);

	EXPECT_EQ(sv.max_size(), 5);
	EXPECT_FALSE(sv.empty());
	ASSERT_EQ(sv.size(), 5);

	unsigned int expected[] = {0, 0, 5, 0, 0};
	for(int i = 0; i < 5; i++)
	{
		EXPECT_EQ(sv[i], expected[i]);
	}
}

TEST(small_vector, InsertBig)
{
	small_vector<unsigned int, 5> sv(5);

	sv.insert(sv.begin() + 2, 5);

	EXPECT_GT(sv.max_size(), 5);
	EXPECT_FALSE(sv.empty());
	ASSERT_EQ(sv.size(), 6);

	unsigned int expected[] = {0, 0, 5, 0, 0, 0};
	for(int i = 0; i < sv.size(); i++)
	{
		EXPECT_EQ(sv[i], expected[i]);
	}
}

TEST(small_vector, EraseSingle)
{
	small_vector<unsigned int, 5> sv{1,2,3,4,5};

	EXPECT_EQ(sv.max_size(), 5);
	EXPECT_FALSE(sv.empty());
	ASSERT_EQ(sv.size(), 5);

	sv.erase(sv.begin() + 2);
	ASSERT_EQ(sv.size(), 4);

	unsigned int expected[] = {1,2,4,5};
	for(int i = 0; i < sv.size(); i++)
	{
		EXPECT_EQ(sv[i], expected[i]);
	}
}

TEST(small_vector, EraseRange)
{
	small_vector<unsigned int, 5> sv{1,2,3,4,5};

	EXPECT_EQ(sv.max_size(), 5);
	EXPECT_FALSE(sv.empty());
	ASSERT_EQ(sv.size(), 5);

	sv.erase(sv.begin() + 1, sv.begin() + 3);
	ASSERT_EQ(sv.size(), 3);

	unsigned int expected[] = {1,4,5};
	for(int i = 0; i < sv.size(); i++)
	{
		EXPECT_EQ(sv[i], expected[i]);
	}
}
