import sys
import os
import select
import pandas as pd
import numpy as np

import argparse
import configparser

from keras.models import Sequential
from keras.layers import Dense, Activation

# custom libs
from networklib import NNDict

def compute_reward_vec(data, damageFactor=.01):

	#Kill, death is worth 1 and -1, damage is this times damage factor
	[m,n] = np.shape(data)
	rewardVec = np.zeros(m)

	for i in range(0,m-1):	

		#State S and Sp, initialize r for this transition to 0
		s = data[i,0:32]
		sp = data[i+1,0:32]

		#Damage, kill, death
		agentDamageTaken = sp[2]-s[2]
		opponentDamageTaken = sp[2+16]-s[2+6]

		#Kill, death
		gotKill   = sp[3]<s[3]
		gotKilled = sp[3+16]<s[3]

		r = gotKill - gotKilled + damageFactor * (opponentDamageTaken - agentDamageTaken)
		rewardVec[i] = r

	return rewardVec


#Can call directly by importing batchlearning
def train_model(model, data, gamma, iterations):
	
	[m,n] = np.shape(data)
	rewardVec = compute_reward_vec(data)

	#Cannot pre compute x,y because model has to do new prediction each time.
	for j in range(0,iterations):
		for i in range(0,m-1):			
			print(i/m)
			
			#State S and Sp, action, reward (precomputed in the data file)
			s = data[i,0:32].reshape(-1,32)
			sp = data[i+1,0:32].reshape(-1,32)
			a = int(data[i,32])
			r = rewardVec[i]

			#We want the model to predict that Q(s,a) = r + max(Q(s',a))
			#Then get the curent prediction from state s
			#Update the current prediciton for action a with the target
			target = r + gamma * np.max(model.predict(sp))			#Exactly what we want model to predict for action in state s
			targetVec = model.predict(s)[0]							#Current prediciton of action values in state s
			targetVec[a] = target									#Fill in what we want to predict

			#Fit the model
			model.fit(x=s, y=targetVec.reshape(-1,63), epochs=1, verbose=0)
	return model

#If running from command line!
def main():

	#Argparse
	parser = argparse.ArgumentParser(description='Batch learning for SS228')
	parser.add_argument('--configfile','-p',default = 'config.ini',
	                    help='Specify different config file for different training runs.')
	args = parser.parse_args()
	config = configparser.ConfigParser()
	config.read(args.configfile)

	#Load params from config
	gamma = float(config['NNLearning']['gamma'])
	iterations = int(config['NNLearning']['iterations'])

	#Load data
	logFileName = config['NNLearning']['logFile']
	logFolderName = config['NNLearning']['logFolder']
	logFolderRoot = config['NNLearning']['logFolderRoot']
	logFile = logFolderRoot+'/'+logFolderName+'/'+logFileName

	df = pd.read_csv(logFile, header=None)
	data = df.values
	
	#Load the model architecture from the dictionary
	reward = rewardDict[config['NNLearning']['reward_function']]	#Computes rewards
	model = NNDict[config['NNLearning']['model']]					#NN model
	
	#See if we can load model weights
	weightFileName = config['NNLearning']['weightFile']
	weightFoldername = config['NNLearning']['weightFolder']
	weightFolderRoot = config['NNLearning']['weightFolderRoot']
	weightFile = weightFolderRoot+'/'+weightFoldername+'/'+weightFileName
	
	if os.path.isfile(weightFile):
		model.load_weights(weightFile)
	else:
		print("Not loading weights, file does not exist.")

	#Train the model, return it.
	model = model_learning(model=model, data=data, reward=reward, gamma=gamma, iterations=iterations)
	model.save_weights(weightFile)  		

if __name__ == '__main__':
	main()