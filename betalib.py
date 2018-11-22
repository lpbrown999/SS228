import sys
import select
import pandas as pd
import numpy as np
import scipy.stats

#stateVal should always be concatenation of selfState, oppState

def jumper_animation(stateVal):
	#Define if we are on the ground
	ay = stateVal[1]
	anim = int(stateVal[5])
	onground  = int(stateVal[12] and stateVal[1]<1)

	anim_portion = np.zeros(400) 	  #Approximately 400 animation for now
	if (anim < 400) and (anim >= 0):
		anim_portion[anim] = 1		  #Turn on the theta weight associated with the current animation

	#Just flag of if we are on the ground, and the animation gector
	beta = np.concatenate((np.array([onground]),anim_portion))

	return beta

def jumper_animation_new(stateVal):
	#Agent x,y, animation value
	ax = stateVal[0]
	ay = stateVal[1]
	
	anim = int(stateVal[5])
	onground  = int(stateVal[12] and stateVal[1]<1)

	anim_portion_beta = np.zeros(400) #Approximately for now
	if (anim < 400) and (anim >= 0):
		anim_portion_beta[anim] = 1		  #Turn on the theta weight associated with the current animation

	#x basis functions centered at various locations
	sigma = 10
	x_portion = scipy.stats.norm.pdf(ax,np.array(range(-300,300)),sigma)

	#Assemble all basis functions
	beta = np.concatenate( (np.array([0,onground]),anim_portion_beta, x_portion) )

	return beta


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Remeber to update this dictionary when adding a new beta function #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

betaDict = {"1": jumper_animation,		
			"2": jumper_animation_new}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Remeber to update this dictionary when adding a new beta function #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #