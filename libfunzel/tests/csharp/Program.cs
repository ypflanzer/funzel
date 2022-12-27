
using FunzelSharp;

internal class Program
{
	private static void Main(string[] args)
	{
		var v = new FloatSmallVector(new float[]
				{
							1.0f,
							2.0f,
							3.0f
				});

		var t = Tensor.ones(new uint[]{3, 3, 3});

		t = t * 5;

		Console.WriteLine("Hello, World: " + t[0]);
	}
}
