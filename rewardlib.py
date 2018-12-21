import sys
import select
import pandas as pd
import numpy as np

#curData should always be a vertical concatenation of the current state, next state!
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

rewardDict = {"1": fighter_reward}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Remeber to update this dictionary when adding a new beta function #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #