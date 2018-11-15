import numpy as np
import pandas as pandas

import argparse
import configparser
from betalib import betaDict

def main():
	parser = argparse.ArgumentParser(description='Batch learning for SS228')
	parser.add_argument('--configfile','-p',default = 'config.ini',
	                    help='Specify different config file for different training runs.')
	args = parser.parse_args()

	config = configparser.ConfigParser()
	config.read(args.configfile)

	thetaFolderName = config['BatchLearn']['thetaFolder']
	thetaFolderRootName = config['BatchLearn']['thetaFolderRoot']
	thetaPostName = config['BatchLearn']['thetaOutput']

	beta = betaDict[config['BatchLearn']['beta_function']]

	theta = np.load(thetaFolderRootName+'/'+thetaFolderName+'/'+thetaPostName)
	numActions = int(config['BatchLearn']['numActions'])
	
	#Infer len Beta
	lenBeta = int(len(theta)/numActions)
	theta = theta.reshape(numActions, lenBeta)

	#Simulated beta function 
	stateval = [0,0]

	for a,thetaA in enumerate(theta):

		value = np.dot(thetaA,beta(stateval))
		print(thetaA,value,a)

if __name__ == '__main__':
	main()