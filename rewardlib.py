import sys
import select
import pandas as pd
import numpy as np


def jumper_reward_old(curData):
	#Curdata
	# obtain states for agent 1 and agent 2
	# x,y,%,stocks,self.facing,self.action.value, self.action.frame, invulnerable, hitlag frames . . .
	# hitstunframes, charging smash, jumps left, on ground, x speed, y speed, off stage
	x = curData[0,0]
	y = curData[0,1]

	# difference between next frame and current frame, clamped at 0
	if(y < 0):
		reward = 0
	else:
		#reward = max(0 , (curData[1,1] - curData[0,1])**2)
		reward = (max(0 , y))**2

	# penalty for being on the ground
	if(y < 0.1):
		reward -= 5

	# penaly for deviating away from x
	reward += -abs(x)

	return reward

def jumper_reward(curData):
	
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
	if (abs(ax_s) > 50) and (abs(ax_sp)>abs(ax_s)):
		reward -= 5

	return(reward)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Remeber to update this dictionary when adding a new beta function #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

rewardDict = {"1": jumper_reward_old,
			  "2": jumper_reward,
			  "3": jumper_xbound_reward}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Remeber to update this dictionary when adding a new beta function #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #