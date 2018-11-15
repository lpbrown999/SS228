import sys
import select
import pandas as pd
import numpy as np


# beta functions relating to our y value and x value
def jumper_beta_xy_1(stateVal):

	ax = stateVal[0]
	ay = stateVal[1]

	# cap inverse of x
	if(ax < 0.1):
		invax = 1000
	else:
		invax = 1/ax

	# cap inverse of y
	if(ay < 0.1):
		invay = 1000
	else:
		invay = 1/ay

	# set basis functions
	beta = np.array([ax**2, ay**2, invax, invay])
	
	return beta

# beta functions relating to our y value
def jumper_beta_y(stateVal):

	ay = stateVal[1]

	# cap inverse of y
	if(ay < 0.1):
		invay = 1000
	else:
		invay = 1/ay

	# set basis functions
	beta = np.array([ay**2,invay])

	return beta

def jumper_beta_xy_2(stateVal):

	ax = stateVal[0]
	ay = stateVal[1]

	# cap inverse of x
	if(ax < 0.1):
		invax = 1000
	else:
		invax = 1/ax

	# cap inverse of y
	if(ay < 0.1):
		invay = 1000
	else:
		invay = 1/ay

	# set basis functions
	beta = np.array([ax**3, ay**3, invax, invay])
	
	return beta



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Remeber to update this dictionary when adding a new beta function #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

betaDict = {"1": jumper_beta_xy_1,
			"2": jumper_beta_y,
			"3": jumper_beta_xy_2}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Remeber to update this dictionary when adding a new beta function #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



# example of how to use this dicionary of function

"""
from beta import betaDict

functionHandle = betaDict["3"]

print(functionHandle(4))
"""




