import melee
from melee.enums import Action, Button

import numpy as np
import pandas as pd

import os
import random

#Own library to do batch Q learning
import batchlearning

class SS228agent():
	def __init__(self, dolphin, gamestate, selfPort, opponentPort, logFile, tempLogFile, style, beta, reward, betaLen, alpha, gamma, iterations, batchLearn, explStrat, explParam):
		
		#Self info about game state
		self.gameState = gamestate
		self.selfState = self.gameState.player[selfPort]
		self.oppState  = self.gameState.player[opponentPort]
		self.controller = melee.controller.Controller(port=selfPort, dolphin=dolphin) 
		self.framedata = melee.framedata.FrameData()
		
		#Logger files, array
		self.logFile = logFile
		self.tempLogFile = tempLogFile
		self.logArray = np.array([])

		#Playstyle, beta function
		self.style = style
		self.beta = beta 
		self.reward = reward
		self.betaLen = betaLen

		#Learning Params
		self.alpha = alpha
		self.gamma = gamma
		self.iterations = iterations
		self.batchLearn = batchLearn     #Toggle for updating theta
		self.matchCompl = 0				 #Matches played, gets incirmented at post game score

		#Information for inputs
		self.framesBetweenInputs = 12
		self.framesSinceInput = 12
		self.inputsBetweenCsvLog = 5    
		self.lastAction = 0
		self.buttonSkipList = [Button.BUTTON_MAIN, Button.BUTTON_C, Button.BUTTON_R, Button.BUTTON_L]

		#Define values for sticks,buttons
		self.stickVals = np.linspace(0,1,3)
		self.buttonVals = [0,1]

		#Shape of ndarray of action matrix that we use to map linear indexes of actions to
		self.numStickVals = len(self.stickVals)
		self.actionsShape = (6,self.numStickVals,self.numStickVals)
		self.numActions = np.prod(self.actionsShape)

		#Exploration
		self.explStrat = explStrat
		self.explParam = float(explParam)

	def simple_button_press(self, actionNumber):
		#Take in an action number, unravel it to the action vector
		#(6,3,3) = 54 actions
		# [(x,A,B,Z,L,none) mx my]

		if not self.controller.pipe:
			return
		
		# obtain button press vector from our action number
		buttonPressVec = self.action_to_controller(actionNumber)

		
		#Tilt the sticks
		mx = buttonPressVec[1]
		my = buttonPressVec[2]    
		self.controller.tilt_analog(Button.BUTTON_MAIN, mx, my)
		self.controller.tilt_analog(Button.BUTTON_C,  .5, .5)

		#If statements to handle the abxLz or nothing to press
		skipList = self.buttonSkipList.copy()

		if buttonPressVec[0] == 0:
			self.controller.press_button(Button.BUTTON_X)
			skipList.append(Button.BUTTON_X)
		
		elif buttonPressVec[0] == 1:
			self.controller.press_button(Button.BUTTON_A)
			skipList.append(Button.BUTTON_A)
		
		elif buttonPressVec[0] == 2:
			self.controller.press_button(Button.BUTTON_B)
			skipList.append(Button.BUTTON_B)

		elif buttonPressVec[0] == 3:
			self.controller.press_button(Button.BUTTON_Z)
			skipList.append(Button.BUTTON_Z)

		#Separate if block for L since we need to press 0 not release
		if buttonPressVec[0] == 4:
			self.controller.press_shoulder(Button.BUTTON_L, 1)
			skipList.append(Button.BUTTON_L)
		else:
			self.controller.press_shoulder(Button.BUTTON_L, 0)

		#If buttonPressVec[0] == 5, no button will be pressed

		#Release unpressed buttons
		for item in Button:          
			if item in skipList:
				continue
			else:
				self.controller.release_button(item)
		
	def act(self):
		
		#If it has been enough frames -> we need a new input
		if self.framesSinceInput >= self.framesBetweenInputs:  
			
			if self.style == 'play':				
				#Compute the expected value of each potential action
				betaCurr = self.beta( np.concatenate((np.array(self.selfState.tolist()),np.array(self.oppState.tolist()))) )
				potentialActionValues = np.zeros(self.numActions)
				for potentialAction in range(0,self.numActions):
					potentialActionValues[potentialAction] = np.dot(self.thetaWeights[potentialAction*self.betaLen:(potentialAction+1)*self.betaLen],betaCurr)

				#Choose an action based on exploration strategy
				actionIdx = self.select_action(potentialActionValues)
			
			elif self.style == 'random':
				actionIdx= random.randrange(0,self.numActions-1)
			elif self.style == 'empty':
				actionIdx = 49
			else:
				actionIdx = 49

			#Execute action, reset counter, record action
			self.simple_button_press(actionIdx)
			self.framesSinceInput = 0
			self.lastAction = actionIdx
		
		#Send an empty input on the frame before we do another input
		elif (self.framesSinceInput == self.framesBetweenInputs - 1):
			self.simple_button_press(49)
			self.framesSinceInput += 1

		else:
			self.framesSinceInput += 1
	
	def select_action(self,potentialActionValues):
		
		if self.explStrat == 'softmax':
			lam = self.explParam
			
			#Need to normalize the potential action values.
			if np.linalg.norm(potentialActionValues) > 0:
				potentialActionValues = potentialActionValues/np.linalg.norm(potentialActionValues)
			
			#Need to normalize the probs since only proportional, must sum to 1.
			prob_i = np.exp(lam*potentialActionValues)
			prob_i = prob_i/sum(prob_i)

			#Possible choices
			actions = np.array(range(0,self.numActions))
			actionIdx = np.random.choice(actions,p=prob_i)
			#print("Animation#: ", self.selfState.tolist()[5] ,"Selected: ", actionIdx,"with probability: ", prob_i[actionIdx])
		else:
			actionIdx = potentialActionValues.argmax()
		return actionIdx

	def state_action_logger(self):
		#Update -> only log on the frames we take an action
		#Logs to a temp file, will get concatenated to the main log at every post game match score.
		if self.framesSinceInput == 0:
			combined_state_action = np.concatenate((np.array(self.selfState.tolist()),np.array(self.oppState.tolist()),np.array([self.lastAction])),axis=0)
			if np.size(self.logArray,axis=0) == self.inputsBetweenCsvLog:
				df = pd.DataFrame(self.logArray)
				df.to_csv(self.tempLogFile, mode='a', header=False, index = False)
				self.logArray = combined_state_action
			elif np.size(self.logArray) == 0:
				self.logArray = combined_state_action
			else:
				self.logArray = np.vstack((self.logArray, combined_state_action))
	
	def templog_to_mainlog(self):
		#The state action logger logs to a temp file. This gets called to remove the temp file.
		df = pd.read_csv(self.tempLogFile, header=None)
		df.to_csv(self.logFile, mode='a', header=False, index = False)
		os.remove(self.tempLogFile)

	def update_theta_weights_learning(self,newThetaFile):
		#Compute new theta using global approximation q learning from batchlearning.py. Using the templog.
		df = pd.read_csv(self.tempLogFile, header=None)
		data = df.values
		thetanew = batchlearning.Q_learning_global_approx(data = data, theta = self.thetaWeights, beta=self.beta, reward=self.reward,
														  alpha = self.alpha, gamma = self.gamma, iterations = self.iterations, 
													      numActions = self.numActions, betaLen = self.betaLen)
		#Save the new weights
		np.save(newThetaFile,thetanew)
	
	def update_theta(self, newThetaFile):
		#Updates from file. Warns if cannot update.
		if os.path.isfile(newThetaFile):
			print("Loading new theta file!: ", newThetaFile)
			self.thetaWeights = np.load(newThetaFile)
		else:
			print("Invalid theta (beta weights) for Initializing to 0s: ", newthetaFile)
			self.thetaWeights = np.zeros(beta(np.zeros(32)).size)  
	
	def action_to_controller(self,actionNumber):
		# unravel action number based on action shape array, obtain stick values
		buttonPressVec = np.array(np.unravel_index(actionNumber,self.actionsShape), dtype=np.float)
		buttonPressVec[1:3] = buttonPressVec[1:3]/(self.numStickVals-1)

		return buttonPressVec

	def controller_to_action(self, buttonPressVec):
		# obtain analog stick action number, ravel into action number
		buttonPressVec[1:3] = buttonPressVec[1:3]*(self.numStickVals-1)
		actionNumber = np.ravel_multi_index(buttonPressVec.astype(int),self.actionsShape)

		return actionNumber
	



