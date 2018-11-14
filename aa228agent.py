import melee
from melee.enums import Action, Button
import random

import numpy as np
import pandas as pd
"""
Testing the potential for random actions
"""
class AA228agent():
    def __init__(self, dolphin, gamestate, self_port, opponent_port, log_file):
        
        #Self info about game state
        self.gamestate = gamestate
        self.self_state = self.gamestate.player[self_port]
        self.opp_state  = self.gamestate.player[opponent_port]
        self.log_file = log_file
        self.log_array = np.array([])

        #Self info about ports
        self.self_port = self_port
        self.opp_port = opponent_port

        self.controller = melee.controller.Controller(port=self_port, dolphin=dolphin) 
        self.framedata = melee.framedata.FrameData()

        #Buttons to skip when pressing buttons
        self.button_skip_list = [Button.BUTTON_MAIN, Button.BUTTON_C, Button.BUTTON_L, Button.BUTTON_R]

        #Information for reduced inputs
        self.frames_between_inputs = 12
        self.frames_since_last_input = 12
        self.frames_between_csv_output = 300
        self.last_action = 0

        #Define values for sticks,buttons
        self.stickvals = np.linspace(0,1,3)
        self.buttonvals = [0,1]

        #Shape of ndarray of action matrix that we use to map linear indexes of actions to
        self.num_stickvals = len(self.stickvals)
        self.actions_shape = (self.num_stickvals,self.num_stickvals,2,2,2,2)
        self.num_poss_actions = np.prod(self.actions_shape)

    def simple_button_press(self, button_press_vec):
        #Inputs as np arary 6x1
        #(3**2)*(2**4) = 144 actions
        #mx my L A B X

        mx = button_press_vec[0]
        my = button_press_vec[1]       
        shoulderval = button_press_vec[2]
        a = button_press_vec[3]
        b = button_press_vec[4]
        x = button_press_vec[5]

        if not self.controller.pipe:
            return

        #Tilt the main stick
        self.controller.tilt_analog(Button.BUTTON_MAIN, mx, my)

        #Tilt the control stick
        self.controller.tilt_analog(Button.BUTTON_C,  .5, .5)

        #Press the shoulder
        self.controller.press_shoulder(Button.BUTTON_L, shoulderval)
        self.controller.press_shoulder(Button.BUTTON_R, 0)

        #Press buttons
        for item in Button:
            
            if item in self.button_skip_list:
                continue
            #Press buttons for a,b,x, release all others
            elif (item == Button.BUTTON_A) and (a == 1):
                self.controller.press_button(item)
            elif (item == Button.BUTTON_B) and (b == 1):
                self.controller.press_button(item)
            elif (item == Button.BUTTON_X) and (x == 1):
                self.controller.press_button(item)
            else:
             self.controller.release_button(item)
        
    def act(self):
        #If it has been enough frames -> we need a new input
        if self.frames_since_last_input == self.frames_between_inputs:  
            
            #Linear index of action
            action_lin_idx= random.randrange(0,self.num_poss_actions-1)

            #map the linear index to the button list, input it to simple button press
            button_list = np.array( np.unravel_index(action_lin_idx,self.actions_shape), dtype=np.float)
            button_list[0:2] = button_list[0:2]/(self.num_stickvals-1)  #0 or 1 is good for the last 4 entries, but need to map stick indexes 0,1,2 
            self.simple_button_press(button_list)

            #Reset counter, recored the last action taken
            self.frames_since_last_input == 0
            self.last_action = action_lin_idx
        else:
            self.frames_since_last_input += 1

    def state_action_logger(self):
        #Current state of self and other bot
        #Our state, opponent state, our last action
        # print((np.array(self.self_state.tolist()),np.array(self.opp_state.tolist()),np.array(self.last_action)))
        combined_state_action = np.concatenate((np.array(self.self_state.tolist()),np.array(self.opp_state.tolist()),np.array([self.last_action])),axis=0)
        
        #Log the array
        if np.size(self.log_array,axis=0) == self.frames_between_csv_output:
            df = pd.DataFrame(self.log_array)
            df.to_csv(self.log_file, mode='a', header=False, index = False)
            self.log_array = combined_state_action
        elif np.size(self.log_array) == 0:
            self.log_array = combined_state_action
        else:
            self.log_array = np.vstack((self.log_array, combined_state_action))





