namespace funzel
{
public partial class Tensor : global::System.IDisposable
{
	public static Tensor empty(uint[] shape, DTYPE dtype = DTYPE.DFLOAT32, string backend = "")
	{
		return empty(new SizeSmallVector(shape), dtype, backend);
	}
	
	public static Tensor ones(uint[] shape, DTYPE dtype = DTYPE.DFLOAT32, string backend = "")
	{
		return ones(new SizeSmallVector(shape), dtype, backend);
	}

	public static Tensor zeros(uint[] shape, DTYPE dtype = DTYPE.DFLOAT32, string backend = "")
	{
		return zeros(new SizeSmallVector(shape), dtype, backend);
	}

	public override string ToString()
	{
		return toString();
	}

	public static Tensor operator-(Tensor t) => t.mul(-1);
	public static Tensor operator-(Tensor a, Tensor b) => a.sub(b);
	public static Tensor operator+(Tensor a, Tensor b) => a.add(b);
	public static Tensor operator*(Tensor a, Tensor b) => a.matmul(b);
	public static Tensor operator/(Tensor a, Tensor b) => a.div(b);

	public static Tensor operator+(Tensor a, double b) => a.add(b);
	public static Tensor operator-(Tensor a, double b) => a.add(-b);
	public static Tensor operator*(Tensor a, double b) => a.mul(b);
	public static Tensor operator/(Tensor a, double b) => a.mul(1.0/b);

	public static Tensor operator*(double b, Tensor a) => a.mul(b);

	public Tensor this[uint idx]
	{
		get { return this.get(idx); }
	}

	public Tensor this[uint[] idx]
	{
		get { return this.get(new SizeSmallVector(idx)); }
	}

	public static uint[] _(params uint[] v) => v;
}
}
