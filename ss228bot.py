#!/usr/bin/python3
import argparse
import configparser
import os
import signal
import sys
import melee

import numpy as np
import pandas as pd

import shutil
import select

#Own libs
from ss228agent import SS228agent
from networklib import NNDict


##TODO
#Training 1. randomize sequencing. Make a vector of indexes, randomize its order, 
# 		     train in that order. So still have a s,a,r,sp but now they arent ocrrelated
#	      2. Have 2 networks when training -> the one used to make predictions for the actions, 
#			 then train the new network separately, then return the new network.
#		  3. Provide the state as a history of states, actions. 
#New method: continuous actions with 2 networks
#
#General: cleanup agent 2 to make same as agent 1
def enter_detected():
	# poll stdin with 0 seconds for timeout
	i,o,e = select.select([sys.stdin],[],[],0)
	if(i):
		return True
	else:
		return False

#Argument, config file parsing.
parser = argparse.ArgumentParser(description='Example of libmelee in action')
parser.add_argument('--configfile','-p',default = 'config.ini',
					help='Specify different config file for different bot runs.')
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.configfile)

#Set up the ports tyepes based on the mode
if config['Agent1']['actor'] == 'AI':
	port1Type = melee.enums.ControllerType.STANDARD    #Bot
elif config['Agent1']['actor'] == 'Human':
	port1Type = melee.enums.ControllerType.GCN_ADAPTER
else:
	print("Exiting, actor not defined.")
	sys.exit()

if config['Agent2']['actor'] == 'AI':
	port2Type = melee.enums.ControllerType.STANDARD    #Bot
elif config['Agent2']['actor'] == 'Human':
	port2Type = melee.enums.ControllerType.GCN_ADAPTER
else:
	print("Exiting, actor not defined.")
	sys.exit()

#Create our Dolphin object. This will be the primary object that we will interface with. 
dolphin = melee.dolphin.Dolphin(ai_port=2,
								opponent_port=1,
								opponent_type=port1Type,
								logger=None)

gamestate = melee.gamestate.GameState(dolphin)
def signal_handler(signal, frame):
	dolphin.terminate()
	print("Shutting down cleanly...")
	sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)
print("Dolphin connected.")

#Initialize agents
agent1 = None
agent2 = None

#Global for both agents for learning
weightFolderRootName= config['NNLearning']['weightFolderRoot']
logFolderRootName 	= config['NNLearning']['logFolderRoot']
alpha 				= float(config['NNLearning']['alpha'])
gamma 				= float(config['NNLearning']['gamma'])
iterations 			= int(config['NNLearning']['iterations'])

if port1Type == melee.enums.ControllerType.STANDARD: 
	
	#Feed the game info, log file, style, beta funciton, exploration to the agent. 
	#Then update the agents theta parameters to be the initial theta file.
	logFile     = logFolderRootName+'/'+config['Agent1']['logFolder']+'/'+config['Agent1']['logFile']
	tempLogFile = logFolderRootName+'/'+config['Agent1']['logFolder']+'/temp'+config['Agent1']['logFile']

	#Load model, load weights into model
	model1 = NNDict[config['Agent1']['model']]

	weightFoldername = config['Agent1']['weightFolder']
	weightFileName = config['Agent1']['weightFile']
	weightFile1 = weightFolderRootName+'/'+weightFoldername+'/'+weightFileName
	if os.path.isfile(weightFile1):
		model1.load_weights(weightFile1)
	else:
		print("Not loading weights for agent1, file does not exist.")
	
	agent1 = SS228agent(dolphin=dolphin, gamestate=gamestate, selfPort=1, opponentPort=2, 
						logFile=logFile, tempLogFile=tempLogFile, style=config['Agent1']['style'],
						model=model1, alpha=alpha, gamma=gamma, iterations=iterations, 
						learn=config['Agent1']['updateWeight'],
						explStrat = config['Agent1']['explStrat'], explParam = config['Agent1']['explParam'] ) 

	agent1.controller.connect()
	print("Agent1 controller connected.")

if port2Type == melee.enums.ControllerType.STANDARD:

	#Feed the game info, log file, style, beta funciton, exploration to the agent. 
	#Then update the agents theta parameters to be the initial theta file.
	logFile     = logFolderRootName+'/'+config['Agent2']['logFolder']+'/'+config['Agent2']['logFile']
	tempLogFile = logFolderRootName+'/'+config['Agent2']['logFolder']+'/temp'+config['Agent2']['logFile']
	beta = betaDict[config['Agent2']['beta_function']]
	reward = rewardDict[config['Agent2']['reward_function']]
	betaLen = len(beta(np.zeros(32)))
	thetaFile2 = thetaFolderRootName+'/'+config['Agent2']['thetaFolder']+'/'+config['Agent2']['thetaFile'] 
	  
	agent2 = SS228agent(dolphin = dolphin, gamestate = gamestate, selfPort = 2, opponentPort = 1,                         
						logFile = logFile, tempLogFile = tempLogFile,  style = config['Agent2']['style'],                    
						beta = beta, reward = reward, betaLen = betaLen,                                                 
						alpha = alpha, gamma = gamma, iterations = iterations, batchLearn = config['Agent2']['updateTheta'],            
						explStrat = config['Agent2']['explStrat'], explParam = config['Agent2']['explParam'] )               

	agent2.update_theta(thetaFile2)
	agent2.controller.connect()
	print("Agent2 controller connected.")

#Main loop
while True:

	#Checking for exit request
	if enter_detected():
		print('Keyboard break detected: cleaning pipes, flushing empty inputs.')
		if agent1:
			agent1.controller.empty_input()
			agent1.controller.flush()
		if agent2:
			agent2.controller.empty_input()
			agent2.controller.flush()

		dolphin.terminate()
		sys.exit(0)

	#Step to next frame
	gamestate.step()
	if(gamestate.processingtime * 1000 > 12):
		print("WARNING: Last frame took " + str(gamestate.processingtime*1000) + "ms to process.")

	#In Game -> act
	if gamestate.menu_state == melee.enums.Menu.IN_GAME:
		if agent1:
			agent1.act()
			agent1.state_action_logger()
		if agent2:
			agent2.act()
			agent2.state_action_logger()

	#If we're at the character select screen, choose our character
	elif gamestate.menu_state == melee.enums.Menu.CHARACTER_SELECT:
		if agent1:
			melee.menuhelper.choosecharacter(character=melee.enums.Character.CPTFALCON,
											 gamestate=gamestate,
											 port=1,
											 opponent_port=2,
											 controller=agent1.controller,
											 swag=False,
											 start=True)
		if agent2:
			melee.menuhelper.choosecharacter(character=melee.enums.Character.CPTFALCON,
											 gamestate=gamestate,
											 port=2,
											 opponent_port=1,
											 controller=agent2.controller,
											 swag=False,
											 start=True)

	#If we're at the postgame scores screen -> batch learnand update theta if desired, concat templog into main log, skip postgame.
	#Check to make sure we only do the batch learning process once per post game.
	elif gamestate.menu_state == melee.enums.Menu.POSTGAME_SCORES:
		
		if agent1:
			if os.path.isfile(agent1.tempLogFile):				#If the temp log still exists, we need to concatenate it to main log and learning
				if agent1.learn == 'True':			
					print("Updating agent 1 model.")
					agent1.matchCompl += 1																			#Incrmt the matches complete parameter of agent 1.
					newWeightFileA1 = os.path.splitext(weightFile1)[0]+'_a1_match'+str(agent1.matchCompl)+'.h5'		#Construct new weight file name
					agent1.learn_new_weights(newWeightsFile=newWeightFileA1)			    						#Learn new weights
				agent1.templog_to_mainlog()
			melee.menuhelper.skippostgame(controller=agent1.controller)
		
		if agent2:
			if os.path.isfile(agent2.tempLogFile):				
				if agent2.batchLearn == 'True':	
					print("Updating Agent2 theta weights, saving theta file.")
					agent2.matchCompl += 1																		
					newThetaFileA2 = os.path.splitext(thetaFile2)[0]+'_a2_match'+str(agent2.matchCompl)+'.npy'		
					agent2.update_theta_weights_learning(newThetaFile = newThetaFileA2)						
					agent2.update_theta(newThetaFile = newThetaFileA2)				  
				
				agent2.templog_to_mainlog()

			melee.menuhelper.skippostgame(controller=agent2.controller)

	#If we're at the stage select screen, choose a stage
	elif gamestate.menu_state == melee.enums.Menu.STAGE_SELECT:
		if agent1 and agent2:
			melee.menuhelper.choosestage(stage=melee.enums.Stage.FINAL_DESTINATION,
							gamestate=gamestate,
							controller=agent1.controller)
			agent2.controller.empty_input()

		elif agent1:
			melee.menuhelper.choosestage(stage=melee.enums.Stage.FINAL_DESTINATION,
							gamestate=gamestate,
							controller=agent1.controller)
		elif agent2:
			melee.menuhelper.choosestage(stage=melee.enums.Stage.FINAL_DESTINATION,
							gamestate=gamestate,
							controller=agent2.controller)

	#Issue queued button commands
	if agent1:
		agent1.controller.flush()
	if agent2:
		agent2.controller.flush()
