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
	
	[m,n] = np.shape(data)
	for j in range(0,iterations):
		for i in range(0,m-1):
			#print('Iter Pctg:', round(j/iterations,3), 'Cur iter pctg:', round(i/(m-1),3))
			
			#State S and Sp
			s_sp = np.vstack((data[i,:],data[i+1,:]))

			#Skip updating theta for actions that did not effect our characters animation
			animation_s = s_sp[0,5]
			animation_frame_s = s_sp[0,6]
			animation_sp = s_sp[1,5]
			animation_frame_sp = s_sp[1,6]
			if (animation_s == animation_sp) and (animation_frame_sp > animation_frame_s):
				continue

			# Action taken at state s, reward from states and sp
			r = reward(s_sp)			
			action = int(s_sp[0,-1])

			# Beta evaluated at s, sp
			betaCur = beta(s_sp[0,:])
			betaNext = beta(s_sp[1,:])
			
			#Maximization term
			term2 = np.zeros(numActions)
			for maxa in range(0,numActions):
				term2[maxa] = np.dot(theta[maxa*betaLen:(maxa+1)*betaLen],betaNext)
			theta[action*betaLen:(action+1)*betaLen] += alpha*(r + gamma*max(term2) - np.dot(theta[action*betaLen:(action+1)*betaLen],betaCur))*betaCur
	
	#Return updated theta
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


