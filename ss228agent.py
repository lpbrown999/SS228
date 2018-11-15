import melee
from melee.enums import Action, Button
import random

import numpy as np
import pandas as pd

class SS228agent():
    def __init__(self, dolphin, gamestate, selfPort, opponentPort, logFile, thetaWeights):
        
        #Self info about game state
        self.gameState = gamestate
        self.selfState = self.gameState.player[selfPort]
        self.oppState  = self.gameState.player[opponentPort]
        self.logFile = logFile
        self.logArray = np.array([])

        self.controller = melee.controller.Controller(port=selfPort, dolphin=dolphin) 
        self.framedata = melee.framedata.FrameData()

        #Buttons to skip when pressing buttons
        self.buttonSkipList = [Button.BUTTON_MAIN, Button.BUTTON_C, Button.BUTTON_R]

        #Information for reduced inputs
        self.framesBetweenInputs = 12
        self.framesSinceInput = 12
        self.framesBetweenCsvLog = 300
        self.lastAction = 0

        #Define values for sticks,buttons
        self.stickVals = np.linspace(0,1,3)
        self.buttonVals = [0,1]

        #Shape of ndarray of action matrix that we use to map linear indexes of actions to
        self.numStickVals = len(self.stickVals)
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
        print(buttonPressVec,self.gameState.frame)
        
        #Tilt the sticks
        mx = buttonPressVec[0]
        my = buttonPressVec[1]    
        self.controller.tilt_analog(Button.BUTTON_MAIN, mx, my)
        self.controller.tilt_analog(Button.BUTTON_C,  .5, .5)

        #If statements to handle the abxLz or nothing to press
        #print(self.framesSinceInput,self.framesBetweenInputs,buttonPressVec[2])

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
        
    def act(self,mode='random'):
        
        #If it has been enough frames -> we need a new input
        if self.framesSinceInput >= self.framesBetweenInputs:  
            
            #Maybe a better way to handle this? separate functions?
            #Choose an action based on the mode!
            if mode == 'random':
                actionIdx= random.randrange(0,self.numActions-1)
            elif mode == 'jumper':
                #Greedy
                betaCurr = self.jumper_beta()
                bestActionTerms  = np.zeros(self.numActions)
                for maxa in range(0,self.numActions):
                    bestActionTerms[maxa] = np.dot(self.thetaWeights[maxa*self.betaLen:(maxa+1)*self.betaLen],betaCurr)
                actionIdx = bestActionTerms.argmax()    #Linear index of the best action
            elif mode == 'empty':
                actionIdx = 24

            #Execute action, reset counter, record action
            self.simple_button_press(actionIdx)
            self.framesSinceInput = 0
            self.lastAction = actionIdx
        
        #Send an empty input on the frame before we do another input
        elif (self.framesSinceInput == self.framesBetweenInputs - 1):
            self.simple_button_press(24)
            self.framesSinceInput += 1

        else:
            self.framesSinceInput += 1

    def jumper_beta(self):
        currSelfState = np.array(self.selfState.tolist())
        ax = currSelfState[0]
        ay = currSelfState[1]

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
        if np.size(self.logArray,axis=0) == self.framesBetweenCsvLog:
            df = pd.DataFrame(self.logArray)
            df.to_csv(self.logFile, mode='a', header=False, index = False)
            self.logArray = combined_state_action
        elif np.size(self.logArray) == 0:
            self.logArray = combined_state_action
        else:
            self.logArray = np.vstack((self.logArray, combined_state_action))




