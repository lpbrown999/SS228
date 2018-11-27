import sys
import select
import pandas as pd
import numpy as np


#curData should always be a vertical concatenation of the current state, next state!
#Only used for batch learning

def jumper_reward(curData):

	s = curData[0,:]
	sp = curData[1,:]

	reward = 0

	#On the ground in state s, sp.
	onground_s  = int(s[12] and s[1]<1)
	onground_sp = int(sp[12] and sp[1]<1)

	#Reward if we leave the ground between s, sp
	#Penalize if we fail to leave the ground
	if (onground_s == 1) and (onground_sp == 0): #We left the ground
		reward += 1
	elif (onground_s == 1) and (onground_sp == 1): #we were on ground and failed to leave the ground -> heavy penalty
		reward -= 5
	else:										#we were not on the ground
		reward += 0

	return(reward)

def jumper_xbound_reward(curData):
	reward = 0

	s = curData[0,:]
	sp = curData[1,:]

	#On the ground in state s, sp. 1 if on ground and actually on stage, not on platform
	onground_s  = int(s[12] and s[1]<1)
	onground_sp = int(sp[12] and sp[1]<1)
	if (onground_s == 1) and (onground_sp == 0): #We left the ground
		reward += 1
	elif (onground_s == 1) and (onground_sp == 1): #we were on ground and failed to leave the ground -> heavy penalty
		reward -= 5
	else:										#we were not on the ground
		reward += 0

	#Penalty for nearing the stage edge. Center of stage is 0
	ax_s = s[0]
	ax_sp = sp[0]

	#Penalize if we are more than 50 units from center stage and we move further away
	#Reward for moving back to center stage
	if (abs(ax_s) > 50):
		if (abs(ax_sp)>abs(ax_s)):
			reward -= (abs(ax_sp)-50)		#Linear ramping penalty for moving away
		elif (abs(ax_sp)<abs(ax_s)):
			reward += 1

	return(reward)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Remeber to update this dictionary when adding a new beta function #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

rewardDict = {"1": jumper_reward,
			  "2": jumper_xbound_reward}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Remeber to update this dictionary when adding a new beta function #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #