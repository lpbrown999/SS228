# SmashBot for AA2288 - Decision Making Under Uncertainty
###### A Perceptron Q Learning SSBM Captain Falcon Agent

### FAQ

1. **What?**

    A repo of a reinforcement learning (Perceptron Q learning) that trains an agent to play Captain Falcon effectively in the video game "Super Smash Brothers Melee".

2. **Why?**
    
    We chose this as our final project for AA228: Decision Making under Uncertainty at Stanford University, Fall 2018 (http://web.stanford.edu/class/aa228/).

## Setup Steps - From Smashbot:

1. Install libmelee, a Python 3 API for interacting with Dolphin and Melee. It's in pip. On most OS's, the command will look like:
`sudo pip3 install melee`.

2. Install the Dolphin version here:
https://github.com/altf4/dolphin/tree/memorywatcher-rebased
This contains an important update to allow Dolphin to be able to read information from Melee. Ensure you clone the memeorywatcher-rebased branch, and additionally if you are on MacOS you will need to install wxmac 3.1.1 via homebrew before building.

3. Make sure you're running Melee v1.02 NTSC. Other versions will not work.

4. If you want to play interactively with or against the AI, you'll probably want a GameCube Adapter, available on Amazon here: https://www.amazon.com/Super-Smash-GameCube-Adapter-Wii-U/dp/B00L3LQ1FI

5. If you're using a GameCube Adapter, make sure to install the drivers / confugure the udev rules, as described here:
https://wiki.dolphin-emu.org/index.php?title=How_to_use_the_Official_GameCube_Controller_Adapter_for_Wii_U_in_Dolphin

6. Apply the latest `Melee Netplay Community Settings` Gecko Code. It's available by default in Dolphin 5.0. SmashBot will NOT work properly without this. (Long story) You will need to enable cheat codes in Dolphin by choosing `Config->General Tab->Enable Cheats` Then right click on the Melee game at the Dolphin home screen and go to `Properties->Gecko Codes` to find the Gecko Code list.

7. Apply `Press Y to toggle frozen stages` Gecko Code. If you want to play on Pokemon Stadium, use the frozen version.

7. Configure the bot as desired in config.ini, and make necessary changes in the script file (stage, character, etc).

8. Run `ss228bot.py`

9. Pray you do not clog the system pipes when quitting. For best experience, press enter on terminal to quit the bot, use cmd+Q to quit dolphin, and reset controllers when restarting dolphin.
