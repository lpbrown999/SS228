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
def NN_learning(model, data, reward, gamma, iterations):
	
	[m,n] = np.shape(data)
	for j in range(0,iterations):
		for i in range(5,m-1):			
			print(i/m)
			
			#State S and Sp
			s = data[i,:-1].reshape(-1,32)
			sp = data[i+1,:-1].reshape(-1,32)
			s_sp = np.vstack((s,sp))

			#Action
			a = int(data[i,-1])
			ap = int(data[i+1,-1])

			#Reward for state s to sp transition
			r = reward(s_sp)			

			#We want the model to predict that Q(s,a) = r + max(Q(s',a))
			#Then get the curent prediction from state s
			#Update the current prediciton for action a with the target
			target = r + gamma * np.max(model.predict(sp))
			targetVec = model.predict(s)[0]
			targetVec[a] = target

			#Fit the model
			model.fit(x=s, y=targetVec.reshape(-1,63), epochs=1, verbose=0)

	return model

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
	
	df = pd.read_csv('logs/oldLogs/Agent1LogRandOvernight.csv', header=None)
	data = df.values

	#Params
	reward = rewardDict['3']
	gamma = .95
	iterations = 1

	NN_learning(model = model, data=data, reward=reward, gamma=gamma, iterations=iterations)

if __name__ == '__main__':
	main()