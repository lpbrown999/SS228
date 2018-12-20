import sys
import select
import pandas as pd
import numpy as np

import argparse
import configparser

# custom lib
from rewardlib import rewardDict

#To do the kills and deaths -> track the index of the action to assign the reward to.
#To do state evolution, also need to track the index of the action we want to assign the reward to.

def compute_reward_vec(data, reward):

	[m,n] = np.shape(data)
	rewardVec = np.zeros(m)

	##Going to see if the agent can learn for itself without hand feeding it state evolution
	# #For deaths
	# actionHistoryLen = 20
	# actionIdxHistory = [0]*actionHistoryLen

	# #For kills
	# lastDamagingActionIdx = None

	# #For general state evolution -> used to track action, betacur. Now track the row of the action.
	# lastMeaningfulAction = 

	for i in range(5,m-1):			#Skip the startup of the game -> 1 second

		#State S and Sp
		s = data[i,:-1].reshape(-1,32)
		sp = data[i+1,:-1].reshape(-1,32)
		s_sp = np.vstack((s,sp))

		#Reward from on going state evolution. Identify if did damage, died, got kill.
		r = reward(s_sp)	

		#Fill in reward vec	
		rewardVec[i] = r

	return rewardVec

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
	reward = rewardDict[config['RewardNN']['reward_function']]

	#Log file locations to read from
	inputFileName = config['RewardNN']['logFile']
	inputFolderName = config['RewardNN']['logFolder']
	inputFolderRootName = config['RewardNN']['logFolderRoot']
	filename = inputFolderRootName+'/'+inputFolderName+'/'+inputFileName

	df = pd.read_csv(filename, header=None)
	data = df.values

	#Compute reward, add to last column (0-31 are state, 32 is action, 33 is rewards), save
	rewardVec = compute_reward_vec(data=data, reward=reward)
	df['33'] = rewardVec
	df.to_csv(filename, mode='w', header=False, index = False)

if __name__ == '__main__':
	main()


