dofile("common.lua")
local funzel = require "LuaFunzel"

if #arg < 3 then
	print("Usage: ", arg[0], " <factor> <input> <output>")
	return 0
end

local factor = tonumber(arg[1])
local input = arg[2]
local output = arg[3]

print("Multiplying image with: ", factor)

print("Loading ", input)
local inImg = funzel.image.load(input, funzel.HWC, funzel.DFLOAT32)

print("Transforming...")
inImg:mul_(factor * 255)

print("Writing ", output)
funzel.image.save(inImg:astype(funzel.DUINT8), output)
