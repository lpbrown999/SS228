import sys
import select
import pandas as pd
import numpy as np

import argparse
import configparser

# custom lib
from rewardlib import rewardDict


from keras.models import Sequential
from keras.layers import Dense, Activation



#Can call directly by importing batchlearning
def NN_learning(model, data, reward, alpha, gamma, iterations):
	
	killReward = 100	#Assigned to last damaging action before kill
	deathReward = -20	#Assigned to all actions in sequence of actions that caued death.

	#Incase not(inSameAnimSequence) does not go off on first run through.
	action = 0
	betaCur = beta(data[0,:])

	#For assigning rewards to kills
	lastDamagingAction = None
	lastDamagingBetaS = beta(data[0,:])
	lastDamagingBetaSp = beta(data[0,:])

	#For assigning penalties to deaths
	actionHistoryLen = 20
	actionHistory = [0]*actionHistoryLen
	betaHistoryS =  [0]*actionHistoryLen
	betaHistorySp = [0]*actionHistoryLen

	[m,n] = np.shape(data)
	for j in range(0,iterations):
		for i in range(5,m-1):			#Skip the startup of the game -> 1 second

			#State S and Sp
			s_sp = np.vstack((data[i,:],data[i+1,:]))

			#Assign credit for state evolution (tm Jeremy Crowley patent pending) to actions that caused a new animation sequence, rather than skipping over entirely (remember 11/26 convo about falcon kick across stage (eggnog))
			animation_s = s_sp[0,5]
			animation_frame_s = s_sp[0,6]
			animation_sp = s_sp[1,5]
			animation_frame_sp = s_sp[1,6]
			inSameAnimSequence = ((animation_s == animation_sp) and (animation_frame_sp > animation_frame_s))
			
			if not(inSameAnimSequence):
				action = int(s_sp[0,-1])
				betaCur = beta(s_sp[0,:])

			#Reward from on going state evolution. Identify if did damage, died, got kill.
			r = reward(s_sp)			
			betaNext = beta(s_sp[1,:])						# Beta evaluated at s, sp

			#Maximization term / Normal Q learning
			term2 = np.zeros(numActions)
			for maxa in range(0,numActions):
				term2[maxa] = np.dot(theta[maxa*betaLen:(maxa+1)*betaLen],betaNext)
			theta[action*betaLen:(action+1)*betaLen] += alpha*(r + gamma*max(term2) - np.dot(theta[action*betaLen:(action+1)*betaLen],betaCur))*betaCur
			

			##Managing extra rewards / penalites for kills / deaths
			#Insert into action history for when we get a kill or die
			oppPctgS  = s_sp[0,2+16]
			oppPctgSp = s_sp[1,2+16]
			ourStockS  = s_sp[0,3]
			ourStockSp = s_sp[1,3]
			oppStockS  = s_sp[0,3+16]
			oppStockSp = s_sp[1,3+16]

			died = 0
			gotKill = 0	
			if (oppPctgSp-oppPctgS) > 1:			#If we did damage, update the last damaging action.
				lastDamagingAction = action
				lastDamagingBetaS  = betaCur
				lastDamagingBetaSp = betaNext
			if (ourStockSp<ourStockS):
				died = 1
			if (oppStockSp<oppStockS):
				gotKill = 1

			#If we got a kill and have dealt damage, assign kill reward to the last damaging action.
			if gotKill and (lastDamagingAction != None):
				term2 = np.zeros(numActions)
				for maxa in range(0,numActions):
					term2[maxa] = np.dot(theta[maxa*betaLen:(maxa+1)*betaLen],lastDamagingBetaSp)

				theta[lastDamagingAction*betaLen:(lastDamagingAction+1)*betaLen] += alpha*(killReward + gamma*max(term2) - np.dot(theta[lastDamagingAction*betaLen:(lastDamagingAction+1)*betaLen],lastDamagingBetaS))*lastDamagingBetaS

			#Manage lists for death
			actionHistory.pop()
			betaHistoryS.pop()
			betaHistorySp.pop()
			actionHistory.insert(0,action)
			betaHistoryS.insert(0,betaCur)
			betaHistorySp.insert(0,betaNext)

			#Assign penalties to the action history that lead to a death.
			if died:
				for indAOld,actionOld in enumerate(actionHistory):	
					betaCurDied  = betaHistoryS[indAOld]
					betaNextDied = betaHistorySp[indAOld]

					if  (type(betaCurDied) is np.ndarray) and (type(betaNextDied) is np.ndarray): #Catch edge case of dying before list populated
						
						term2 = np.zeros(numActions)
						for maxa in range(0,numActions):
							term2[maxa] = np.dot(theta[maxa*betaLen:(maxa+1)*betaLen],betaNextDied)
						theta[actionOld*betaLen:(actionOld+1)*betaLen] += alpha*(deathReward + gamma*max(term2) - np.dot(theta[actionOld*betaLen:(actionOld+1)*betaLen],betaCurDied))*betaCurDied

	return theta

#If running from command line!
def main():

	#Make the model
	input_dim = 32
	output_dim = 63

	model = Sequential()
	model.add(Dense(32,    activation='relu', input_dim=input_dim))	#Input state vec layer
	model.add(Dense(32*32, activation='relu'))
	model.add(Dense(output_dim,    activation='linear'))				#Output action layer
	model.compile(loss='mse', optimizer='adam', metrics=['mae'])
	
	#Load data

	NN_learning(model = model, data=data, reward=reward, alpha=alpha, gamma=gamma, iterations=iterations)

if __name__ == '__main__':
	main()