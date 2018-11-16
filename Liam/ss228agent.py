# standard lib
import melee
from melee.enums import Action, Button
import random
import numpy as np
import pandas as pd

import random
# custom lib


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

        #Information from global Q learning
        self.thetaWeights = thetaWeights
        self.betaLen = int(len(self.thetaWeights)/self.numActions)

        #Pass these as more config file shit
        self.explorationStrategy = 'softmax'
        self.explorationParam = 9

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
            
            #Change this to a switch?
            if self.style == 'random':
                actionIdx= random.randrange(0,self.numActions-1)
            
            elif self.style == 'jumper':
                
                #Compute the expected value of each potential action
                betaCurr = self.beta( np.concatenate((np.array(self.selfState.tolist()),np.array(self.oppState.tolist()))) )
                potentialActionValues = np.zeros(self.numActions)
                for potentialAction in range(0,self.numActions):
                    potentialActionValues[potentialAction] = np.dot(self.thetaWeights[potentialAction*self.betaLen:(potentialAction+1)*self.betaLen],betaCurr)

                #Choose an action based on exploration strategy
                actionIdx = self.select_action(potentialActionValues)
                            
            elif self.style == "forcejump":

            	jump = np.array([0, 0.5, 0.5])
            	actionIdx = self.controller_to_action(jump)

            elif self.style == 'empty':
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

    def state_action_logger(self):
        #Update -> only log on the frames we take an action
        #Since the logger is called after the action function,
        #This is when self.framesSinceInput == 0
        #Can revert back by removing just this if statement
        if self.framesSinceInput == 0:
            combined_state_action = np.concatenate((np.array(self.selfState.tolist()),np.array(self.oppState.tolist()),np.array([self.lastAction])),axis=0)
            #Log the array
            if np.size(self.logArray,axis=0) == self.inputsBetweenCsvLog:
                df = pd.DataFrame(self.logArray)
                df.to_csv(self.logFile, mode='a', header=False, index = False)
                self.logArray = combined_state_action
            elif np.size(self.logArray) == 0:
                self.logArray = combined_state_action
            else:
                self.logArray = np.vstack((self.logArray, combined_state_action))

    def select_action(self,potentialActionValues):
        
        if self.explorationStrategy == 'softmax':
            lam = self.explorationParam #SOFTMAX PARAM -> since norming action values to 1 so we dont get infd, need to have this pretty high
            
            #Need to normalize the potential action values so we dont get INFd out of our minds
            potentialActionValues = potentialActionValues/np.linalg.norm(potentialActionValues)
            prob_i = np.exp(lam*potentialActionValues)
            #print(sum(prob_i),prob_i)

            #Normalize probs since only proportional from exp
            prob_i = prob_i/sum(prob_i)

            actions = np.array(range(0,len(prob_i)))
            actionIdx = np.random.choice(actions,p=prob_i)
            print("Animation#: ", self.selfState.tolist()[5] ,"Selected: ", actionIdx,"with probability: ", prob_i[actionIdx])

        else:
            actionIdx = potentialActionValues.argmax()

        return actionIdx

    # obtain controller inputs given action number
    def action_to_controller(self,actionNumber):

        # unravel action number based on action shape array
        buttonPressVec = np.array(np.unravel_index(actionNumber,self.actionsShape), dtype=np.float)

        # obtain analog stick values
        buttonPressVec[1:3] = buttonPressVec[1:3]/(self.numStickVals-1)

        return buttonPressVec

    # obtain action value from controller input
    def controller_to_action(self, buttonPressVec):

        # obtain analog stick action number
        buttonPressVec[1:3] = buttonPressVec[1:3]*(self.numStickVals-1)

        # ravel button press vector back to action number
        actionNumber = np.ravel_multi_index(buttonPressVec.astype(int),self.actionsShape)

        return actionNumber
        



