#!/usr/bin/python3
import argparse
import os
import signal
import sys
import melee

import numpy as np
import pandas as pd

import shutil
import select

# from esagent import ESAgent -> could make Agent 2 this agent.
from aa228agent import AA228agent

#TODO
#Logging of ACTIONS
#Map each posisble state to a state #
#Map each aciton to an action #
#Might need to discretize observations
#Omit things from our observations (who cares about speed etc..)

#Q learning
#can back propogate our reward to "nearest neighbor states" similar to section 4.5.
#eligibility traces

def enter_detected():
    # poll stdin with 0 seconds for timeout
    i,o,e = select.select([sys.stdin],[],[],0)
    if(i):
        return True
    else:
        return False

def check_port(value):
    ivalue = int(value)
    if ivalue < 1 or ivalue > 4:
         raise argparse.ArgumentTypeError("%s is an invalid controller port. \
         Must be 1, 2, 3, or 4." % value)
    return ivalue

#Rethink how we start up the script. I.e.:
#python smashbot.py -1 'Human' -2 'AI'
parser = argparse.ArgumentParser(description='Example of libmelee in action')
parser.add_argument('--mode','-m',type=int,default = 1,
                    help='Different Modes:\n \
                    1 - Human on Port 1, AI on Port 2. \n \
                    2 - AI on Port 1 and AI on Port 2. \n \
                    3 - Can be added for future use.')
parser.add_argument('--logging','-l', action='store_true',
                    help='Logging of Gamestates')
args = parser.parse_args()

#Setup the ports based on what mode we selected
#   Options here are:
#   GCN_ADAPTER will use your WiiU adapter for live human-controlled play
#   UNPLUGGED is pretty obvious what it means
#   STANDARD is a named pipe input (bot)
if args.mode == 1:    #Human vs AI
    port1_type = melee.enums.ControllerType.GCN_ADAPTER #Human
    port2_type = melee.enums.ControllerType.STANDARD    #Bot
elif args.mode == 2:    #AI vs AI
    port1_type = melee.enums.ControllerType.STANDARD
    port2_type = melee.enums.ControllerType.STANDARD
else:
    print("Exiting, mode set to 0 which is improper usage")
    sys.exit()

#Create our Dolphin object. This will be the primary object that we will interface with. 
dolphin = melee.dolphin.Dolphin(ai_port=2,
                                opponent_port=1,
                                opponent_type=port1_type,
                                logger=None)
gamestate = melee.gamestate.GameState(dolphin)
def signal_handler(signal, frame):
    dolphin.terminate()
    print("Shutting down cleanly...")
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)
print("Dolphing connected.")

#Initialize all the agents that we have designated are bots by the port type, and connect the controllers
agent1 = None
agent2 = None
if port1_type == melee.enums.ControllerType.STANDARD:
    agent1 = AA228agent(dolphin = dolphin, gamestate = gamestate, self_port = 1, opponent_port = 2, 
                        logFile = 'agent1_logNew.csv', thetaWeights = np.load('thetaNew.npy'))
    agent1.controller.connect()
    print("Agent1 controller connected.")
if port2_type == melee.enums.ControllerType.STANDARD:
    agent2 = AA228agent(dolphin = dolphin, gamestate = gamestate, self_port = 2, opponent_port = 1, 
                        logFile = 'agent2_logNew.csv', thetaWeights = np.load('thetaNew.npy'))
    agent2.controller.connect()
    print("Agent2 controller connected.")

#Main loop
while True:
    #print(gamestate.frame)
    if enter_detected():
        print('Keyboard break detected: cleaning pipes, flusshing empty inputs.')
        if agent1:
            agent1.controller.empty_input()
            agent1.controller.flush()
        if agent2:
            agent2.controller.empty_input()
            agent2.controller.flush()

        dolphin.terminate()
        sys.exit(0)

    gamestate.step()
    if(gamestate.processingtime * 1000 > 12):
        print("WARNING: Last frame took " + str(gamestate.processingtime*1000) + "ms to process.")

    #In game -> act
    if gamestate.menu_state == melee.enums.Menu.IN_GAME:
        
        if agent1:
            agent1.jumper()
            agent1.state_action_logger()
        if agent2:
            p = 0
            #agent2.act()
            #agent2.state_action_logger()

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

    #If we're at the postgame scores screen, spam START
    elif gamestate.menu_state == melee.enums.Menu.POSTGAME_SCORES:
        if agent1:
            melee.menuhelper.skippostgame(controller=agent1.controller)
        if agent2:
            melee.menuhelper.skippostgame(controller=agent2.controller)

    #If we're at the stage select screen, choose a stage
    elif gamestate.menu_state == melee.enums.Menu.STAGE_SELECT:
    	#FINAL_DESTINATION
    	#BATTLEFIELD
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
