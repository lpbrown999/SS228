import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt

def main():
	logs = ['logs/Agent1Logs/overnightFriday.csv', 'logs/Agent1Logs/daySaturday.csv']
	fig1, ax1 = plt.subplots()

	for log in logs:
		df = pd.read_csv(log, header = None)
		data = df.values
		aStock = data[:,3]
		oStock = data[:,3+16]

		winLoss = []
		cumWinPctg = []

		while True:

			#Indexes where each agent has 0 stocks.
			#Find indexes and check if there are none -> if so break b/c reached end of data set
			aidx0 = np.where(aStock == 0)[0]
			oidx0 = np.where(oStock == 0)[0]
			if (aidx0.size == 0) and (oidx0.size ==0):	#Finished all games
				break
			elif aidx0.size == 0:						#No more games where agent loses
				aidx0 = np.inf
				oidx0 = oidx0[0]	
			elif oidx0.size == 0:						#No more games where agent won
				oidx0 = np.inf
				aidx0 = aidx0[0]	
			else:										#Normal
				aidx0 = aidx0[0]
				oidx0 = oidx0[0]
			
			#Want first occurance of 0 stocks in the data, if at this point agent has 0 stocks it was a loss.
			idx0 = min(aidx0,oidx0)
			if aStock[idx0] == 0:
				winLoss.append(0)
			else:
				winLoss.append(1)

			cumWinPctg.append(sum(winLoss)/len(winLoss))

			#Remove all data before this recorded entry. 
			aStock = aStock[idx0+1:]
			oStock = oStock[idx0+1:]

			#Find start of next match
			a4idx = np.where((aStock==4)&(oStock==4))[0]
			if (a4idx.size == 0):
				break										#No more start to games
			a4idx = a4idx[0]
			aStock = aStock[a4idx:]
			oStock = oStock[a4idx:]

		ax1.plot( np.array(range(0,len(cumWinPctg))) , cumWinPctg, label = log)
	ax1.legend()
	ax1.grid()

	plt.show()

	#Find the first time someone hits 0 stocks
	#Find the both stock 0, return first tuple, first time this happened, and look at the state before it.
	#lastMomentIdx = np.where((aStock==0)&(oStock==0))[0][0]-1
	#print(aStock[lastMomentIdx], oStock[lastMomentIdx])

if '__name__':
	main()