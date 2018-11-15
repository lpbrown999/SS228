import sys
import select
import pandas as pd
import numpy as np

import argparse
import configparser

# custom lib
from betalib import betaDict


#functionHandle = betaDict["3"]


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

beta = betaDict[config['BatchLearn']['beta_function']]


def get_jumper_reward(curData):

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

	# penalty
	if(y < 0.1):
		reward -= 5

	reward += -abs(x)

	return reward

def global_approx(dfVals, theta, numActions, betaLen):

	[m,n] = np.shape(dfVals)

	for i in range(0,m-1):
		print(i)

		# calculate reward for jumper
		data = np.vstack((dfVals[i,:],dfVals[i+1,:]))
		reward = get_jumper_reward(data)

		# extract action from data
		action = int(dfVals[i,-1])

		# caluclate maximum value for dot(theta,beta)
		term2 = np.zeros(numActions)
		betaCur = beta(dfVals[i,:])
		betaNext = beta(dfVals[i+1,:])
		
		# find action to maximize 	
		for maxa in range(0,numActions):
			term2[maxa] = np.dot(theta[maxa*betaLen:(maxa+1)*betaLen],betaNext)

		# update spliced theta
		theta[action*betaLen:(action+1)*betaLen] += alpha*(reward + gamma*max(term2) - np.dot(theta[action*betaLen:(action+1)*betaLen],betaCur))*betaCur

		# normalize certain value
		if(np.linalg.norm(theta) != 0):
			thetaRatio = ((1e15)*len(theta))/np.linalg.norm(theta)
		else:
			thetaRatio = 1

		theta = theta*(thetaRatio)

	return theta

def main():

	inputFileName = config['BatchLearn']['logFile']
	inputFolderName = config['BatchLearn']['logFolder']
	inputFolderRootName = config['BatchLearn']['logFolderRoot']

	thetaPriorName = config['BatchLearn']['thetaPrior']
	thetaPostName = config['BatchLearn']['thetaOutput']

	thetaFolderName = config['BatchLearn']['thetaFolder']
	thetaFolderRootName = config['BatchLearn']['thetaFolderRoot']

	df = pd.read_csv(inputFolderRootName+'/'+inputFolderName+'/'+inputFileName, header=None)
	dfVals = df.values

	betaLen = len(beta(dfVals[0,:])) 	# obtain beta length

	# create new theta or grab from previous theta
	if config['BatchLearn']['thetaPrior'] == 'none':
		theta = np.zeros(numActions*betaLen)
	else:
		theta = np.load(thetaFolderRootName+'/'+thetaFolderName+'/'+thetaPriorName)

	# perform global approximation with theta
	theta = global_approx(dfVals, theta, numActions, betaLen)

	# save theta
	np.save(thetaFolderRootName+'/'+thetaFolderName+'/'+thetaPostName,theta)

if __name__ == '__main__':
	main()


