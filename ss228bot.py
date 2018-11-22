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
from betalib import betaDict

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
print("Dolphing connected.")

#Initialize agents
agent1 = None
agent2 = None

thetaFolderRootName = config['BatchLearn']['thetaFolderRoot']
logFolderRootName = config['BatchLearn']['logFolderRoot']

if port1Type == melee.enums.ControllerType.STANDARD: #Agent 1 is a bot
    
    logFile   = logFolderRootName+'/'+config['Agent1']['logFolder']+'/'+config['Agent1']['logFile']
    beta = betaDict[config['Agent1']['beta_function']]
    thetaFile = thetaFolderRootName+'/'+config['Agent1']['thetaFolder']+'/'+config['Agent1']['thetaFile']
    
    #Try to load theta for agent 1. If we cannot, initialize to 0s of same size as beta.
    if os.path.isfile(thetaFile):
        print("Loading theta (beta weights) for agent1: ", thetaFile)
        theta1 = np.load(thetaFile)
    else:
        print("Invalid theta (beta weights) for agent1. Initializing to 0s.: ", thetaFile)
        theta1 = np.zeros(beta(np.zeros(32)).size)                     

    agent1 = SS228agent(dolphin = dolphin, gamestate = gamestate, selfPort = 1, opponentPort = 2,               #info about game
                        logFile = logFile, style = config['Agent1']['style'],                                   #log file, style
                        thetaWeights = theta1,  beta = beta,                                                    #weights, beta
                        explStrat = config['Agent1']['explStrat'], explParam = config['Agent1']['explParam'] )  #exploration strategy
    agent1.controller.connect()
    print("Agent1 controller connected.")

if port2Type == melee.enums.ControllerType.STANDARD:

    logFile   = logFolderRootName+'/'+config['Agent2']['logFolder']+'/'+config['Agent2']['logFile']
    beta = betaDict[config['Agent2']['beta_function']]
    thetaFile = thetaFolderRootName+'/'+config['Agent2']['thetaFolder']+'/'+config['Agent2']['thetaFile']
    
    if os.path.isfile(thetaFile):
        print("Loading theta (beta weights) for agent2: ", thetaFile)
        theta2 = np.load(thetaFile)
    else:
        print("Invalid theta (beta weights) for agent1. Initializing to 0s.: ", thetaFile)
        theta2 = np.zeros(beta(np.zeros(32)).size)                     

    agent2 = SS228agent(dolphin = dolphin, gamestate = gamestate, selfPort = 2, opponentPort = 1,               #info about game
                        logFile = logFile, style = config['Agent2']['style'],                                   #log file, style
                        thetaWeights = theta2,  beta = beta,                                                    #weights, beta
                        explStrat = config['Agent2']['explStrat'], explParam = config['Agent2']['explParam'] )  #exploration strategy
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
            #agent1.state_action_logger()
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

    #If we're at the postgame scores screen -> PROCESS THETAS if desired, load new thetas into agent, concat the logs.
    #Then spam start
    elif gamestate.menu_state == melee.enums.Menu.POSTGAME_SCORES:
        if agent1:
            melee.menuhelper.skippostgame(controller=agent1.controller)
        if agent2:
            melee.menuhelper.skippostgame(controller=agent2.controller)

    #If we're at the stage select screen, choose a stage
    elif gamestate.menu_state == melee.enums.Menu.STAGE_SELECT:
    	#FINAL_DESTINATION
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
