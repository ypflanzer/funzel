
using FunzelSharp;
using static FunzelSharp.Tensor;

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

		var t = Tensor.ones(_(2, 2));

		t = t * 5;

		Console.WriteLine("Hello, World: " + t[0]);
	}
}
