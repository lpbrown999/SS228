import numpy as np
import random
def main():
	dim1 = 3;
	dim2 = 3;
	dim3 = 3;
	dim4 = 3;
	#Buttons are either 0 or 1
	dim5 = 2;
	dim6 = 2;
	dim7 = 2;
	dim8 = 2;
	shape = (dim1,dim2,dim3,dim4,dim5,dim6,dim7,dim8)

	numactions = np.prod(shape)
	print(numactions)
	b = np.ravel_multi_index((1,1,1,1,1,1,1,1),shape)
	c = np.unravel_index(numactions-1,shape)
	print(list(c))
	print(b)

if __name__ == '__main__':
	print('wtf')
	main()