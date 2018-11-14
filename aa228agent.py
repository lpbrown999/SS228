import melee
from melee.enums import Action, Button
import random

import numpy as np
import pandas as pd
"""
Testing the potential for random actions
"""
class AA228agent():
    def __init__(self, dolphin, gamestate, self_port, opponent_port, startTF, log_file):
        
        #Self info about game state
        self.gamestate = gamestate
        self.self_state = self.gamestate.player[self_port]
        self.opp_state  = self.gamestate.player[opponent_port]
        self.log_file = log_file
        self.log_array = np.array([])

        #Self info about ports
        self.self_port = self_port
        self.opp_port = opponent_port
        self.startTF = startTF

        self.controller = melee.controller.Controller(port=self_port, dolphin=dolphin) 
        self.framedata = melee.framedata.FrameData()

        #Buttons to skip when pressing buttons
        self.button_skip_list = [Button.BUTTON_MAIN, Button.BUTTON_C, Button.BUTTON_L, Button.BUTTON_R]

        #Define values for sticks,buttons
        self.stickvals = np.linspace(0,1,3)
        self.buttonvals = [0,1]

        #Shape of ndarray of action matrix that we use to map linear indexes of actions to
        self.num_stickvals = len(self.stickvals)
        self.actions_shape = (self.num_stickvals,self.num_stickvals,2,2,2,2)
        self.num_poss_actions = np.prod(self.actions_shape)

    def simple_button_press(self, button_press_vec):
        #Inputs as np arary 6x1
        #mx my L A B X

        mx = button_press_vec[0]
        my = button_press_vec[1]
        #cx = button_press_vec[2]
        #cy = button_press_vec[3]        
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
        #Empty input
        button_press_vec = np.array([.5,.5,0,0,0,0])
        #(3**2)*(2**4) = 144 actions

        rand_action_lin_idx= random.randrange(0,self.num_poss_actions-1)
        button_list = np.array( np.unravel_index(rand_action_lin_idx,self.actions_shape), dtype=np.float)
        button_list[0:2] = button_list[0:2]/(self.num_stickvals-1)  #0 or 1 is good for the last 4 entries, but need to map stick indexes 0,1,2 
                                                                    #to 0 to 1 -> divide first 4 by numstickvals -1
        #percent_chance_press_a = .05
        #if random.random() < percent_chance_press_a:
        #    button_press_vec[6] = 1

        self.simple_button_press(button_list)
        self.last_action = button_list          #Record last action so we can put into the log file
        self.jumper_logger()

    def jumper(self):
        #Empty input
        button_press_vec = np.array([.5,.5,.5,.5,0,0,0,0])
        self.jumper_logger()


    def jumper_logger(self):
        #Current state of self and other bot
        self_state_current = np.array(self.self_state.tolist())
        opp_state_current  = np.array(self.opp_state.tolist())
        combined_state_action = np.concatenate((np.array(self.self_state.tolist()),np.array(self.opp_state.tolist()),self.last_action),axis=0)
        #print(cat_state)
        
        #Log the array
        if np.size(self.log_array,axis=0) == 100:
            df = pd.DataFrame(self.log_array)
            df.to_csv(self.log_file, mode='a', header=False, index = False)
            self.log_array = combined_state_action
        elif np.size(self.log_array) == 0:
            self.log_array = combined_state_action
        else:
            self.log_array = np.vstack((self.log_array, combined_state_action))


    def full_logger(self):
        pass





