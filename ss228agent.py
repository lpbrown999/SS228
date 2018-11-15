# standard lib
import melee
from melee.enums import Action, Button
import random
import numpy as np
import pandas as pd

import random
# custom lib
# from betalib import betaDict


class SS228agent():
    def __init__(self, dolphin, gamestate, selfPort, opponentPort, logFile, thetaWeights, style, beta):
        
        #Self info about game state
        self.gameState = gamestate
        self.selfState = self.gameState.player[selfPort]
        self.oppState  = self.gameState.player[opponentPort]
        self.logFile = logFile
        self.logArray = np.array([])

        self.controller = melee.controller.Controller(port=selfPort, dolphin=dolphin) 
        self.framedata = melee.framedata.FrameData()

        #Playstyle, beta function
        self.style = style
        self.beta = beta 

        #Information for inputs
        self.framesBetweenInputs = 12
        self.framesSinceInput = 12
        self.framesBetweenCsvLog = 300
        self.lastAction = 0
        self.buttonSkipList = [Button.BUTTON_MAIN, Button.BUTTON_C, Button.BUTTON_R, Button.BUTTON_L]

        #Define values for sticks,buttons
        self.stickVals = np.linspace(0,1,3)
        self.buttonVals = [0,1]

        #Shape of ndarray of action matrix that we use to map linear indexes of actions to
        self.numStickVals = len(self.stickVals)
        self.actionsShape = (self.numStickVals,self.numStickVals,6)
        self.numActions = np.prod(self.actionsShape)

        #Information for global Q learning
        self.thetaWeights = thetaWeights
        self.betaLen = len(self.beta(np.array(self.selfState.tolist())))

    def simple_button_press(self, actionNumber):
        #Take in an action number, unravel it to the action vector
        #(3*3*6) = 54 actions
        #mx my [abxLz none]

        if not self.controller.pipe:
            return
        
        # obtain button press vector from our action number
        buttonPressVec = self.action_to_controller(actionNumber)
        print(buttonPressVec)
        
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
            skipList.append(Button.BUTTON_L)
        else:
            self.controller.press_shoulder(Button.BUTTON_L, 0)
        
        if buttonPressVec[2] == 4:
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
        if self.framesSinceInput >= self.framesBetweenInputs:  
            
            #Maybe a better way to handle this? separate functions?
            #Choose an action based on the style!
            if self.style == 'random':
                actionIdx= random.randrange(0,self.numActions-1)
            
            elif self.style == 'jumper':
                #Eps Greedy
                if random.random() < .1:
                    actionIdx= random.randrange(0,self.numActions-1)
                else:
                    betaCurr = self.beta(np.array(self.selfState.tolist()))
                    bestActionTerms  = np.zeros(self.numActions)
                    for maxa in range(0,self.numActions):
                        bestActionTerms[maxa] = np.dot(self.thetaWeights[maxa*self.betaLen:(maxa+1)*self.betaLen],betaCurr)
                    print(bestActionTerms)
                    actionIdx = bestActionTerms.argmax()    #Linear index of the best action
                    print(actionIdx)                
                    print(bestActionTerms[actionIdx])

            elif self.style == 'empty':
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

    # obtain controller inputs given action number
    def action_to_controller(self,actionNumber):

        # unravel action number based on action shape array
        buttonPressVec = np.array(np.unravel_index(actionNumber,self.actionsShape), dtype=np.float)

        # obtain analog stick values
        buttonPressVec[0:2] = buttonPressVec[0:2]/(self.numStickVals-1)

        return buttonPressVec

    # obtain action value from controller input
    def controller_to_action(self, buttonPressVec):

        # obtain analog stick action number
        buttonPressVec[0:2] = buttonPressVec[0:2]*(self.numStickVals-1)

        # ravel button press vector back to action number
        actionNumber = np.ravel_multi_index(buttonPressVec.astype(int),self.actionsShape)

        return actionNumber
        



