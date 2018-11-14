import sys
from pynput import keyboard



def on_release(key):
	if key == keyboard.Key.ctrl:

		print("ctrl detected, exiting")
		# Stop listener
		return False

# Collect events until released
print("Before listener")
with keyboard.Listener(on_release=on_release) as listener:
	listener.join()

print("After listener")

