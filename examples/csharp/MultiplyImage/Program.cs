
internal class Program
{
	private static void Main(string[] args)
	{
		if(args.Length < 3)
		{
			Console.WriteLine("Usage: program <factor> <input> <output>");
			return;
		}

		var factor = Convert.ToDouble(args[0]);
		var input = args[1];
		var output = args[2];

		Console.WriteLine("Multiplying image with: {0}", factor);

		Console.WriteLine("Loading {0}", input);
		var inImg = funzel.image.load(input, funzel.image.CHANNEL_ORDER.HWC, funzel.DTYPE.DFLOAT32);

		Console.WriteLine("Transforming...");
		inImg.mul_(factor * 255.0);

		Console.WriteLine("Writing {0}", output);
		funzel.image.save(inImg.astype(funzel.DTYPE.DUINT8), output);
	}
}