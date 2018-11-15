import sys
import select
import pandas as pd
import numpy as np

# Note, all these y's are almost meaningless because it spends most of its time on the ground
# and then 0**2 = 0 -> no way for it to train these weights. Need to augment these?

# Beta functions need to be chosen to have ACTIVATION at important states -> return high value
# in states we care about. Currently they are returning 0 when we are in the most critical state (on the ground)
# Idea: beta = sqrt(10-ay) -> gives high activation when on ground, decays upward. Does better than 1/y

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

def jumper_beta_new(stateVal):
	#Curdata
	# obtain states for agent 1 and agent 2
	# x,y,%,stocks,self.facing,self.action.value, self.action.frame, invulnerable, hitlag frames . . .
	# hitstunframes, charging smash, jumps left, on ground, x speed, y speed, off stage

	ay = stateVal[1]
	beta = np.array([1, ay, np.sign(ay)*(ay**2), np.sqrt(max(0,5-ay))])
	
	return beta



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Remeber to update this dictionary when adding a new beta function #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

betaDict = {"1": jumper_beta_xy_1,
			"2": jumper_beta_y,
			"3": jumper_beta_xy_2,
			"4": jumper_beta_new}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Remeber to update this dictionary when adding a new beta function #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



# example of how to use this dicionary of function

"""
from beta import betaDict

functionHandle = betaDict["3"]

print(functionHandle(4))
"""




