import scipy.stats
import numpy as np 
import time

t1 = time.time()
iterations = 1000
for i in range(0,iterations):
	ybasis = np.array(range(-15,60))
	ya = 0
	basis = scipy.stats.norm.pdf(ya,ybasis, 3)
	otherbasis = np.array([ya,0,1])
	fullbasis = np.concatenate((basis,otherbasis))
t2 = time.time()

reqspeed = 16/1000
print("Time to run", t2-t1, "Per iterations:",(t2-t1)/1000, "Pctg req speed", (t2-t1)/(iterations*reqspeed) )