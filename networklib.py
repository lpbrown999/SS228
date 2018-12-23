import sys
import select
import pandas as pd
import numpy as np

import argparse
import configparser

from keras.models import Sequential
from keras.layers import Dense, Activation


parser = argparse.ArgumentParser(description='Batch learning for SS228')
parser.add_argument('--configfile','-p',default = 'config.ini',
                    help='Specify different config file for different training runs.')
args = parser.parse_args()
config = configparser.ConfigParser()
config.read(args.configfile)

inputDim  = int(config['NNLearning']['inputDim'])						#state size
outputDim = int(config['NNLearning']['outputDim'])					#action size


#Model 1
model1 = Sequential()
model1.add(Dense(inputDim,    	   activation='relu', input_dim=inputDim))		#Input state vec layer
model1.add(Dense(60,           activation='relu'))
model1.add(Dense(outputDim,    activation='relu'))				#Output action layer
model1.compile(loss='mse', optimizer='adam', metrics=['mae'])	#Default adam params


#Dictionary of different models
NNDict = 	{"1": model1}