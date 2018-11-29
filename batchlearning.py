import sys
import select
import pandas as pd
import numpy as np

import argparse
import configparser

# custom lib
from betalib import betaDict
from rewardlib import rewardDict


#Can call directly by importing batchlearning
def Q_learning_global_approx(data, theta, beta, reward, alpha, gamma, iterations, numActions, betaLen):
	
	#Incase not(inSameAnimSequence does not go off on same one)
	action = 0
	betaCur = beta(data[0,:])

	#For assigning rewards to kills
	lastDamagingAction = 0
	lastDamagingBetaS = beta(data[0,:])
	lastDamagingBetaSp = beta(data[0,:])

	actionHistoryLen = 20
	actionHistory = [0]*actionHistoryLen
	betaHistoryS =  [0]*actionHistoryLen
	betaHistorySp = [0]*actionHistoryLen

	[m,n] = np.shape(data)
	for j in range(0,iterations):
		for i in range(0,m-1):

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
			if (oppPctgSp-oppPctgS) > 1:			#If we did damage
				lastDamagingAction = action
				lastDamagingBetaS  = betaCur
				lastDamagingBetaSp = betaNext
			if (ourStockSp<ourStockS):
				died = 1
			if (oppStockSp<oppStockS):
				gotKill = 1

			#If we got a kill, update the theta weights for the last damaging action and the kill reward
			if gotKill:
				killReward = 100
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

			#Assign rewards / penalites to action history for kill / death
			if died:
				deathReward = -10
				for indAOld,actionOld in enumerate(actionHistory):	
					
					betaCurDied  = betaHistoryS[indAOld]
					betaNextDied = betaHistorySp[indAOld]
					
					term2 = np.zeros(numActions)
					for maxa in range(0,numActions):
						term2[maxa] = np.dot(theta[maxa*betaLen:(maxa+1)*betaLen],betaNextDied)

					theta[actionOld*betaLen:(actionOld+1)*betaLen] += alpha*(deathReward + gamma*max(term2) - np.dot(theta[actionOld*betaLen:(actionOld+1)*betaLen],betaCurDied))*betaCurDied
				


	return theta

#If running from command line!
def main():

	#parse config
	parser = argparse.ArgumentParser(description='Batch learning for SS228')
	parser.add_argument('--configfile','-p',default = 'config.ini',
	                    help='Specify different config file for different training runs.')
	args = parser.parse_args()
	config = configparser.ConfigParser()
	config.read(args.configfile)

	#Variables, Functions
	numActions = int(config['BatchLearn']['numActions'])
	alpha = float(config['BatchLearn']['alpha'])
	gamma = float(config['BatchLearn']['gamma'])
	iterations = int(config['BatchLearn']['iterations'])
	beta = betaDict[config['BatchLearn']['beta_function']]
	reward = rewardDict[config['BatchLearn']['reward_function']]

	print('Alpha: ', alpha)
	print('Gamma: ', gamma)
	print('Iter: ', iterations)
	print('BetaFunc: '  ,config['BatchLearn']['beta_function'])
	print('RewardFunc: ',config['BatchLearn']['reward_function'])

	#Log file locations
	inputFileName = config['BatchLearn']['logFile']
	inputFolderName = config['BatchLearn']['logFolder']
	inputFolderRootName = config['BatchLearn']['logFolderRoot']

	#Theta file locations
	thetaPriorName = config['BatchLearn']['thetaPrior']
	thetaPostName = config['BatchLearn']['thetaOutput']
	thetaFolderName = config['BatchLearn']['thetaFolder']
	thetaFolderRootName = config['BatchLearn']['thetaFolderRoot']
	
	df = pd.read_csv(inputFolderRootName+'/'+inputFolderName+'/'+inputFileName, header=None)
	data = df.values

	betaLen = len(beta(data[0,:]))
	betaLen = int(len(theta)/numActions)

	# create new theta or grab from previous theta
	if config['BatchLearn']['thetaPrior'] == 'none':
		print('Initialiizing new theta')
		theta = np.zeros(numActions*betaLen)
	else:
		print('Loading previous theta')
		theta = np.load(thetaFolderRootName+'/'+thetaFolderName+'/'+thetaPriorName)
	input("Ok to continue?")

	# perform global approximation with, save to posterior theta name.
	theta = Q_learning_global_approx(data = data, theta = theta, beta = beta, reward = reward, alpha = alpha, gamma = gamma, 
									 iterations = iterations, numActions = numActions, betaLen = betaLen)
	np.save(thetaFolderRootName+'/'+thetaFolderName+'/'+thetaPostName,theta)

if __name__ == '__main__':
	main()


