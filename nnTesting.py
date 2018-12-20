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
def model_learning(model, data, gamma, iterations):
	
	[m,n] = np.shape(data)

	#Cannot pre compute x,y because model has to do new prediction each time.
	for j in range(0,iterations):
		for i in range(5,m-1):			
			print(i/m)
			
			#State S and Sp, action, reward (precomputed in the data file)
			s = data[i,:-2].reshape(-1,32)
			sp = data[i+1,:-2].reshape(-1,32)
			a = int(data[i,-2])
			r = data[i,-1]

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
	gamma = float(config['RewardNN']['gamma'])
	iterations = int(config['RewardNN']['iterations'])

	#Load data, break if reward not there
	inputFileName = config['RewardNN']['logFile']
	inputFolderName = config['RewardNN']['logFolder']
	inputFolderRootName = config['RewardNN']['logFolderRoot']
	filename = inputFolderRootName+'/'+inputFolderName+'/'+inputFileName

	df = pd.read_csv(filename, header=None)
	data = df.values
	if data.shape[1]!=34:
		print("Warning, need to precompute the reward and place in last column of data.")
		sys.exit()

	#Make the model
	input_dim = 32
	output_dim = 63

	model = Sequential()
	model.add(Dense(32,    activation='relu', input_dim=input_dim))		#Input state vec layer
	model.add(Dense(32*32, activation='relu'))
	model.add(Dense(output_dim,    activation='linear'))				#Output action layer
	model.compile(loss='mse', optimizer='adam', metrics=['mae'])

	#Train the model
	model = model_learning(model=model, data=data, gamma=gamma, iterations=iterations)
	model.save('my_model.h5')  		# creates a HDF5 file 'my_model.h5'


if __name__ == '__main__':
	main()