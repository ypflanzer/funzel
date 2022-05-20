import PyFunzel as funzel
import numpy as np

#print("Hello World!")
#t = funzel.Tensor.ones((32))

#print("t:", t)

def MatmulTensorTest():
	a = np.array(
			[[[1, 2, 3],
			[4, 5, 6],
			[7, 8, 9]],

			[[9, 8, 7],
			[6, 5, 4],
			[3, 2, 1]]])

	b = np.array(
			[[[9, 8, 7],
			[6, 5, 4],
			[3, 2, 1]],
			
			[[1, 2, 3],
			[4, 5, 6],
			[7, 8, 9]]])

	print(a@b)

def MatmulTest():
	a = np.array([
		[1.0, 2.0, 3.0],
		[4.0, 5.0, 6.0],
	])

	b = np.array([
		[9.0, 8.0],
		[7.0, 6.0],
		[5.0, 4.0],
	])

	print(a@b)

def BroadcastTestVectorVector():
	a = np.array([
		[1, 2, 3],
		[1, 2, 3],
		[1, 2, 3],
		[1, 2, 3],
	])

	b = np.array([1, 1, 1])
	print(a@b)

MatmulTensorTest()
MatmulTest()

BroadcastTestVectorVector()
