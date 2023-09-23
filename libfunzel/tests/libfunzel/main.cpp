#include "gtest/gtest.h"
#include <funzel/Funzel.hpp>

#include <LoadBackends.hpp>
#include <spdlog/spdlog.h>

int main(int argc, char **argv)
{
	spdlog::set_level(spdlog::level::debug);
	
	LoadBackends();

	::testing::InitGoogleTest(&argc, argv);
	int ret = RUN_ALL_TESTS();
	return ret;
}
