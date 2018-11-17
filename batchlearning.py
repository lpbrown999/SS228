import sys
import select
import pandas as pd
import numpy as np

import argparse
import configparser

# custom lib
from betalib import betaDict
from rewardlib import rewardDict

#parse config
parser = argparse.ArgumentParser(description='Batch learning for SS228')
parser.add_argument('--configfile','-p',default = 'config.ini',
                    help='Specify different config file for different training runs.')
args = parser.parse_args()
config = configparser.ConfigParser()
config.read(args.configfile)

# global variables
numActions = int(config['BatchLearn']['numActions'])
alpha = float(config['BatchLearn']['alpha'])
gamma = float(config['BatchLearn']['gamma'])
print('Alpha: ', alpha)
print('Gamma: ', gamma)

#Functions
print('BetaFunc: '  ,config['BatchLearn']['beta_function'])
print('RewardFunc: ',config['BatchLearn']['reward_function'])
beta = betaDict[config['BatchLearn']['beta_function']]
reward = rewardDict[config['BatchLearn']['reward_function']]
def global_approx(dfVals, theta, numActions, betaLen, iterations):

	[m,n] = np.shape(dfVals)
	for j in range(0,iterations):
		for i in range(0,m-1):
			print('Iter Pctg:', round(j/iterations,3), 'Cur iter pctg:', round(i/(m-1),3))
			
			#State S and Sp
			s_sp = np.vstack((dfVals[i,:],dfVals[i+1,:]))

			#Skip updating theta for actions that did not effect our characters animation
			animation_s = s_sp[0,5]
			animation_frame_s = s_sp[0,6]
			animation_sp = s_sp[1,5]
			animation_frame_sp = s_sp[1,6]
			if (animation_s == animation_sp) and (animation_frame_sp > animation_frame_s):
				continue

			# Action taken at state s,sp, reward from states and sp
			r = reward(s_sp)	#Change to reward lib
			action = int(s_sp[0,-1])
			action_p = int(s_sp[1,-1])

			# Basis functions evaluated at s, sp
			betaCur = beta(s_sp[0,:])
			betaNext = beta(s_sp[1,:])
			
			# #QLEARNING
			# Maximization term
			term2 = np.zeros(numActions)
			for maxa in range(0,numActions):
				term2[maxa] = np.dot(theta[maxa*betaLen:(maxa+1)*betaLen],betaNext)
			theta[action*betaLen:(action+1)*betaLen] += alpha*(r + gamma*max(term2) - np.dot(theta[action*betaLen:(action+1)*betaLen],betaCur))*betaCur
			
			#SARSA
			#theta[action*betaLen:(action+1)*betaLen] += alpha*(r + gamma*np.dot(theta[action_p*betaLen:(action_p+1)*betaLen],betaNext) - np.dot(theta[action*betaLen:(action+1)*betaLen],betaCur))*betaCur

			#Normalize theta to keep bounded since theta abs can diverge -> multiply by arbitray large number
			# if(np.linalg.norm(theta) != 0):
			# 	thetaRatio = ((1e50)*len(theta))/np.linalg.norm(theta)
			# else:
			# 	thetaRatio = 1
		
			#print(max(theta))
			thetaRatio = 1
			theta = theta*(thetaRatio)

	return theta

def main():

	#Read from config file
	inputFileName = config['BatchLearn']['logFile']
	inputFolderName = config['BatchLearn']['logFolder']
	inputFolderRootName = config['BatchLearn']['logFolderRoot']

	thetaPriorName = config['BatchLearn']['thetaPrior']
	thetaPostName = config['BatchLearn']['thetaOutput']

	thetaFolderName = config['BatchLearn']['thetaFolder']
	thetaFolderRootName = config['BatchLearn']['thetaFolderRoot']
	
	iterations = int(config['BatchLearn']['iterations'])

	df = pd.read_csv(inputFolderRootName+'/'+inputFolderName+'/'+inputFileName, header=None)
	dfVals = df.values

	betaLen = len(beta(dfVals[0,:]))

	# create new theta or grab from previous theta
	if config['BatchLearn']['thetaPrior'] == 'none':
		print('Initialiizing new theta')
		theta = np.zeros(numActions*betaLen)
	else:
		print('Loading previous theta')
		theta = np.load(thetaFolderRootName+'/'+thetaFolderName+'/'+thetaPriorName)

	input("Ok to continue?")

	# perform global approximation with theta, save
	theta = global_approx(dfVals, theta, numActions, betaLen, iterations)
	np.save(thetaFolderRootName+'/'+thetaFolderName+'/'+thetaPostName,theta)

if __name__ == '__main__':
	main()


