import melee
from melee.enums import Action, Button
import random

import numpy as np
"""
Testing the potential for random actions
"""
class AA228agent():
    def __init__(self, dolphin, gamestate, self_port, opponent_port, startTF):
        
        #Self info about game state
        self.gamestate = gamestate
        self.self_state = self.gamestate.player[self_port]
        self.opp_state  = self.gamestate.player[opponent_port]
        #self.logCSV = logCSV

        #Self info about ports
        self.self_port = self_port
        self.opp_port = opponent_port
        self.startTF = startTF

        self.controller = melee.controller.Controller(port=self_port, dolphin=dolphin) 
        self.framedata = melee.framedata.FrameData()

        #Define values for sticks,buttons
        self.stickvals = np.linspace(0,1,3)
        self.buttonvals = [0,1]

        #Shape of ndarray of action matrix that we use to map linear indexes of actions to
        self.num_stickvals = len(self.stickvals)
        self.actions_shape = (self.num_stickvals,self.num_stickvals,self.num_stickvals,self.num_stickvals,2,2,2,2)

        #Total number of possible actions
        self.num_poss_actions = np.prod(self.actions_shape)

    def simple_button_press(self, button_press_vec):
        #Inputs as np arary 8x1
        #mx my cx cy L A B X

        mx = button_press_vec[0]
        my = button_press_vec[1]
        cx = button_press_vec[2]
        cy = button_press_vec[3]        
        shoulderval = button_press_vec[4]
        a = button_press_vec[5]
        b = button_press_vec[6]
        x = button_press_vec[7]

        if not self.controller.pipe:
            return

        #Tilt the main stick
        self.controller.tilt_analog(Button.BUTTON_MAIN, mx, my)

        #Tilt the control stick
        self.controller.tilt_analog(Button.BUTTON_C,  cx, cy)

        #Press the shoulder
        self.controller.press_shoulder(Button.BUTTON_L, shoulderval)
        self.controller.press_shoulder(Button.BUTTON_R, 0)

        #Press buttons
        for item in Button:

            #Don't do anything for the main or c-stick
            if item == Button.BUTTON_MAIN:
                continue
            if item == Button.BUTTON_C:
                continue

            #Press buttons for a,b,x, release all others
            if (item == Button.BUTTON_A) and (a == 1):
                self.controller.press_button(item)
            elif (item == Button.BUTTON_B) and (b == 1):
                self.controller.press_button(item)
            elif (item == Button.BUTTON_X) and (x == 1):
                self.controller.press_button(item)
            else:
                self.controller.release_button(item)
    
    def act(self):

        button_press_vec = [.5,.5,.5,.5,0,0,0,0]
        #Absolute random
        #Currently (3**4)*(2**4)=1296 actions
        #Can reduce by removing ability to use c stick-> (3**2)*(2**4) = 144 actions
        #are we allowed to make reward a function of s,s'? could just pass in s, s', and see what damage is done

        rand_action_lin_idx= random.randrange(0,self.num_poss_actions-1)
        button_list = np.array( np.unravel_index(rand_action_lin_idx,self.actions_shape), dtype=np.float)
        button_list[0:4] = button_list[0:4]/(self.num_stickvals-1)  #0 or 1 is good for the last 4 entries, but need to map stick indexes 0,1,2 
                                                                    #to 0 to 1 -> divide first 4 by numstickvals -1
        #percent_chance_press_a = .05
        #if random.random() < percent_chance_press_a:
        #    button_press_vec[6] = 1

        self.simple_button_press(button_list)
        self.reduced_logger()

    def reduced_logger(self):
        self_state_current = self.self_state.tolist()
        opp_state_current  = self.opp_state.tolist() 
        
        #print(self.self_state_current.tolist())
        
        #x,y,%,stocks,self.facing,self.action.value, self.action.frame, invulnerable, hitlag frames, hitstunframes...
        #charging smash, jumps left, on ground, x speed, y speed, off stage

        #x locations
        #blast zone to blast zone is -225 to 225 in x
        #edge of stage is -68 to -68
        
        #y
        #160 is cieling 0 is main stage
        #side plats are 27, 54
        #-111 is bottom blast zone 
        #print(temp[1])
        #print(smashbotx,smashboty,playerx,playery)
    def full_logger(self):
        #Work on this - > embed with in actor
# if logging:
#     log_array = np.array(gamestate.tolist())
#     log_count = 0;

# pre_existing_Q = False;
# if pre_existing_Q:
#     print('pre Q')
# else:    
#     print('no pre Q')

# print(agent1.num_poss_actions)

        #Define logger for the agent
        if logging:
            if log_count == 100:
                df = pd.DataFrame(log_array)
                df.to_csv('logging.csv', mode='a', header=False, index = False)
                log_array = np.array(gamestate.tolist())
                log_count = 0
            else:
                log_array = np.vstack((log_array, gamestate.tolist()))
                log_count +=1
        pass


