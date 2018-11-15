import sys
import select
import pandas as pd
import numpy as np



def get_jumper_reward(curData):


	# obtain states for agent 1 and agent 2
	# x,y,%,stocks,self.facing,self.action.value, self.action.frame, invulnerable, hitlag frames . . .
	# hitstunframes, charging smash, jumps left, on ground, x speed, y speed, off stage
	
	# difference between next frame and current frame, clamped at 0
	if(curData[1,1] < 0 or curData[0,1] < 0):
		reward = 0
	else:
		reward = max(0 , curData[1,1] - curData[0,1])

	if(curData[0,1] < 0.1):
		reward -= 5

	return reward/1000

# compute beta functions based on state
def beta(stateVal):

	ax = stateVal[0]
	ay = stateVal[1]

	if(ax < 0.1):
		invax = 1000
	else:
		invax = 1/ax

	if(ay < 0.1):
		invay = 1000
	else:
		invay = 1/ay

	beta = np.array([ax**2, ay**2, invax, invay])

	return beta




def main():

	if len(sys.argv) != 2:
		raise Exception("usage: python3 jumperbatch.py <infile>.csv")

	inputfilename = sys.argv[1]
	numStatesVars = 16
	numActions = 144

	alpha = 0.05
	gamma = 0.95

	df = pd.read_csv(inputfilename, header=None)
	dfVals = df.values

	# generate theta
	betaLen = len(beta(dfVals[0,:]))
	theta = np.zeros(numActions*betaLen)

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


		theta[action*betaLen:(action+1)*betaLen] += alpha*(reward + gamma*max(term2) - np.dot(theta[action*betaLen:(action+1)*betaLen],betaCur))*betaCur

		if(sum(theta) > 0):
			thetaRatio = (1000*len(theta))/sum(theta)
		else:
			thetaRatio = 1

		theta = theta*(thetaRatio)
		#theta = theta/(max(1, max(theta)))
		#print(theta)
		#print(theta)
	np.save("theta.npy",theta)
	














if __name__ == '__main__':
	main()


