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

	# #On the ground in state s, sp. 1 if on ground and actually on stage, not on platform
	# onground_s  = int(s[12] and s[1]<1)
	# onground_sp = int(sp[12] and sp[1]<1)
	# if (onground_s == 1) and (onground_sp == 0): #We left the ground
	# 	reward += 1
	# elif (onground_s == 1) and (onground_sp == 1): #we were on ground and failed to leave the ground -> heavy penalty
	# 	reward -= 5
	# else:										#we were not on the ground
	# 	reward += 0

	#Penalty for nearing the stage edge. Center of stage is 0
	ax_s = s[0]
	ax_sp = sp[0]

	#Penalize if we are more than 50 units from center stage and we move further away
	#Reward for moving back to center stage
	good_x = 20
	if (abs(ax_s) > good_x):
		if (abs(ax_sp)>=abs(ax_s)):
			reward -= (abs(ax_sp)-good_x)		#Linear ramping penalty for moving away
		elif (abs(ax_sp)<abs(ax_s)):
			reward += 1

	elif (abs(ax_sp)<good_x):
		reward += 1

	return reward

#Only based on damage
def fighter_reward(curData):
	reward = 0
	didDamage = 0

	s = curData[0,:]
	sp = curData[1,:]

	#Agent, opponent percents in state S, Sp
	aPctgS  = s[2]
	aPctgSp = sp[2]

	oPctgS  = s[2+16]
	oPctgSp = sp[2+16]

	aDelPctg = aPctgSp - aPctgS
	oDelPctg = oPctgSp - oPctgS


	#On stage
	aOffStageS = int(s[15])
	aOffStageSp= int(sp[15])


	#Assign reward for doing damage -> ignore doing or taking 1 damge for hoop damage.
	if oDelPctg > 1:
		reward += oDelPctg*np.exp(-0.01*oPctgS)

	#Penalty for taking damage -> no multiplyer
	if aDelPctg > 1:
		reward -= aDelPctg*np.exp(-0.01*aPctgS)

	if aOffStageS and not(aOffStageSp):
		reward += 5

	return reward

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Remeber to update this dictionary when adding a new beta function #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

rewardDict = {"1": jumper_reward,
			  "2": jumper_xbound_reward,
			  "3": fighter_reward}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Remeber to update this dictionary when adding a new beta function #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #