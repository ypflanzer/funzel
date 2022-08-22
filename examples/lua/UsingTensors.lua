dofile("common.lua")
local funzel = require "LuaFunzel"

-- Create tensor with all ones
local t1 = funzel.Tensor.ones({3, 3, 3})

print("t1 * 5 = " .. tostring(t1 * 5))
print("t1[1] = " .. tostring(t1[1]))
print("t1[{1, 1}] = " .. tostring(t1[{1, 1}]))

print("t1.shape = " .. tostring(t1.shape))
print("t1.strides = " .. tostring(t1.strides))
