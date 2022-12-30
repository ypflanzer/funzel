import common
import sys
import PyFunzel as funzel

if len(sys.argv) < 4:
	print("Usage: ", sys.argv[0], " <factor> <input> <output>")
	exit(0)

factor = float(sys.argv[1])
input = sys.argv[2]
output = sys.argv[3]

print("Multiplying image with: ", factor)

print("Loading ", input)
inImg = funzel.load(input, funzel.HWC, funzel.DFLOAT32)

print("Transforming...")
inImg.mul_(factor * 255)

print("Writing ", output)
funzel.save(inImg.astype(funzel.DUINT8), output)
