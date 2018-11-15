import melee
from melee.enums import Action, Button
import random

import numpy as np
import pandas as pd
"""
Testing the potential for random actions
"""
class AA228agent():
    def __init__(self, dolphin, gamestate, self_port, opponent_port, logFile, thetaWeights):
        
        #Self info about game state
        self.gameState = gamestate
        self.selfState = self.gameState.player[self_port]
        self.oppState  = self.gameState.player[opponent_port]
        self.logFile = logFile
        self.log_array = np.array([])

        #Self info about ports
        self.self_port = self_port
        self.opp_port = opponent_port

        self.controller = melee.controller.Controller(port=self_port, dolphin=dolphin) 
        self.framedata = melee.framedata.FrameData()

        #Buttons to skip when pressing buttons
        self.buttonSkipList = [Button.BUTTON_MAIN, Button.BUTTON_C, Button.BUTTON_R]

        #Information for reduced inputs
        self.frames_between_inputs = 12
        self.frames_since_last_input = 12
        self.frames_between_csv_output = 300
        self.lastAction = 0

        #Define values for sticks,buttons
        self.stickvals = np.linspace(0,1,3)
        self.buttonvals = [0,1]

        #Shape of ndarray of action matrix that we use to map linear indexes of actions to
        self.numStickVals = len(self.stickvals)
        self.actionsShape = (self.numStickVals,self.numStickVals,6)
        self.numActions = np.prod(self.actionsShape)

        #Information for global Q learning
        self.thetaWeights = thetaWeights
        self.betaLen = len(self.jumper_beta())

    def simple_button_press(self, actionNumber):
        #Take in an action number, unravel it to the action vector
        #(3*3*6) = 54 actions
        #mx my [abxLz none]

        if not self.controller.pipe:
            return
        
        buttonPressVec = np.array(np.unravel_index(actionNumber,self.actionsShape), dtype=np.float)
        buttonPressVec[0:2] = buttonPressVec[0:2]/(self.numStickVals-1)

        mx = buttonPressVec[0]
        my = buttonPressVec[1]    


        #Tilt the sticks
        self.controller.tilt_analog(Button.BUTTON_MAIN, mx, my)
        self.controller.tilt_analog(Button.BUTTON_C,  .5, .5)

        #If statements to handle the abxLz or nothing to press
        #print(self.frames_since_last_input,self.frames_between_inputs,buttonPressVec[2])

        skipList = self.buttonSkipList.copy()
        if buttonPressVec[2] == 5:
            self.controller.press_shoulder(Button.BUTTON_L, 1)
            skipList.append(Button.BUTTON_Z)
        
        elif buttonPressVec[2] == 4:
            self.controller.press_button(Button.BUTTON_Z)
            skipList.append(Button.BUTTON_Z)

        elif buttonPressVec[2] == 3:
            self.controller.press_button(Button.BUTTON_X)
            skipList.append(Button.BUTTON_X)
        
        elif buttonPressVec[2] == 2:
            self.controller.press_button(Button.BUTTON_B)
            skipList.append(Button.BUTTON_B)

        elif buttonPressVec[2] == 1:
            self.controller.press_button(Button.BUTTON_A)
            skipList.append(Button.BUTTON_A)

        #Release unpressed buttons
        for item in Button:          
            if item in skipList:
                continue
            else:
                self.controller.release_button(item)
        
    def act(self):
        
        #If it has been enough frames -> we need a new input
        #print(self.frames_since_last_input,self.frames_between_inputs)
        #print(self.frames_since_last_input,self.frames_between_inputs)
        if self.frames_since_last_input >= self.frames_between_inputs:  
        
            #Linear index of action
            actionIdx= random.randrange(0,self.numActions-1)
            self.simple_button_press(actionIdx)

            #Reset counter, recored the last action taken
            self.frames_since_last_input = 0
            self.lastAction = actionIdx

        else:
            self.frames_since_last_input += 1
    
    def jumper(self):        

        #If it has been enough frames -> we need a new input
        if self.frames_since_last_input == self.frames_between_inputs:  
            
            #Greedy
            betaCurr = self.jumper_beta()
            bestActionTerms  = np.zeros(self.numActions)
            for maxa in range(0,self.numActions):
                bestActionTerms[maxa] = np.dot(self.thetaWeights[maxa*self.betaLen:(maxa+1)*self.betaLen],betaCurr)

            #Linear index of the best action
            actionIdx = bestActionTerms.argmax()
            self.simple_button_press(actionIdx)

            #Reset counter, recored the last action taken
            self.frames_since_last_input = 0
            self.lastAction = actionIdx

            #print(actionIdx)
        else:
            self.frames_since_last_input += 1

    def jumper_beta(self):
        curr_self_state = np.array(self.selfState.tolist())
        ax = curr_self_state[0]
        ay = curr_self_state[1]

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

    def state_action_logger(self):
        combined_state_action = np.concatenate((np.array(self.selfState.tolist()),np.array(self.oppState.tolist()),np.array([self.lastAction])),axis=0)
        
        #Log the array
        if np.size(self.log_array,axis=0) == self.frames_between_csv_output:
            df = pd.DataFrame(self.log_array)
            df.to_csv(self.logFile, mode='a', header=False, index = False)
            self.log_array = combined_state_action
        elif np.size(self.log_array) == 0:
            self.log_array = combined_state_action
        else:
            self.log_array = np.vstack((self.log_array, combined_state_action))





