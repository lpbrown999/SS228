import sys
from pynput import keyboard
import select

# returns true if enter detected from keyboard
def enter_detected():

	# poll stdin with 0 seconds for timeout
    i,o,e = select.select([sys.stdin],[],[],0)
    if(i):
    	return True
    else:
    	return False


"""
while(1):
	
	if(enter_detected()):
		print("keyboard detected")
		break
"""
