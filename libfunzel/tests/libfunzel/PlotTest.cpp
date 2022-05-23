#include <gtest/gtest.h>
#include <funzel/Plot.hpp>
#include <cmath>

using namespace funzel;

TEST(Plot, Const)
{
	Plot p;
	p.title("This is a plot!");

	size_t num = 100;
	Tensor t1 = Tensor::ones({num, 2});
	Tensor t2 = Tensor::ones(10);

	p.plot(t2, "Ones");

	t2.mul_(0.8);
	p.plot(t2, "Zeropointeights");

	for(int i = 0; i < t1.shape[0]; i++)
	{
		t1[i][0].set(i*0.1);
		t1[i][1].set(std::sin(i*0.1)*2.0f);
	}

	p.plot(t1, "T1")->shape("lines").color("red");
	p.save("PlotConst.png");
}

TEST(Plot, Linear)
{
	Plot p;
	p.title("This is a plot!");

	float num = 100;
	Tensor t = funzel::linspace(
		Tensor({2}, {0.0f, 0.0f}),
		Tensor({2}, {num*0.1f, 1.0f}),
		num
	);

	p.plot(t, "Linear")->shape("lines").color("red");
	p.save("PlotLinear.png");
}

TEST(Plot, LinearArange)
{
	Plot p;
	p.title("This is a plot!");

	float num = 100;
	Tensor t = funzel::arange(0, num, 12.5);

	p.plot(t, "Linear Arange")->shape("lines").color("red");
	p.plot(t, "")->shape("points").color("blue");

	p.save("PlotArange.png");
}
