import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt

def main():
	logs = ['logs/Agent1Logs/overnightFriday.csv', 'logs/Agent1Logs/daySaturday.csv',
			'logs/Agent1Logs/nightSunday.csv','logs/Agent1Logs/dayMonday.csv']

	fig1, ax1 = plt.subplots()

	for log in logs:
		df = pd.read_csv(log, header = None)
		data = df.values

		aStock = data[:,3]
		oStock = data[:,3+16]

		aDamage = data[:,2]
		oDamage = data[:,2+16]

		winLoss = []
		cumWinPctg = []

		#By game
		agentTotalDamage = []
		opponentTotalDamage = []

		cumADamage = []
		cumODamage = []

		while True:

			#Find subset of data that is the "game"
			startGameIdx = np.where(aStock == 4)[0]
			if startGameIdx.size == 0:
				break
			else:
				startGameIdx = startGameIdx[0]

			#End of game indexes where each agent has 0 stocks.
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
			endGameIdx = min(aidx0,oidx0)

			#SUBSETS FOR THIS GAME
			aStockSubset = aStock[startGameIdx:endGameIdx+1]
			oStockSubset = oStock[startGameIdx:endGameIdx+1]
			aDamageSubset = aDamage[startGameIdx:endGameIdx+1]
			oDamageSubset = oDamage[startGameIdx:endGameIdx+1]

			#Agent
			agentTotalDamage.append(0)
			for stock in range(3,-1,-1):
				agentNewLifeIdx = np.where(aStockSubset==stock)[0]
				
				if agentNewLifeIdx.size == 0:						#No more deaths
					agentTotalDamage[-1] += aDamageSubset[-1]
					break
				
				agentTotalDamage[-1] += aDamageSubset[agentNewLifeIdx[0]-1]
			cumADamage.append(sum(agentTotalDamage)/len(agentTotalDamage))
			
			#Opponent
			opponentTotalDamage.append(0)
			for stock in range(3,-1,-1):
				oppNewLifeIdx = np.where(oStockSubset==stock)[0]
				
				if oppNewLifeIdx.size == 0:						#No more deaths
					opponentTotalDamage[-1] += oDamageSubset[-1]
					break

				opponentTotalDamage[-1] += oDamageSubset[oppNewLifeIdx[0]-1]
			cumODamage.append(sum(opponentTotalDamage)/len(opponentTotalDamage))


			#Remove this game
			aStock = aStock[endGameIdx+1:]
			oStock = oStock[endGameIdx+1:]
			aDamage = aDamage[endGameIdx+1:]
			oDamage = oDamage[endGameIdx+1:]

			#Find start of next match
			a4idx = np.where((aStock==4)&(oStock==4))[0]
			if (a4idx.size == 0):
				break										#No more start to games
			a4idx = a4idx[0]
			aStock = aStock[a4idx:]
			oStock = oStock[a4idx:]
			aDamage = aDamage[a4idx:]
			oDamage = oDamage[a4idx:]

		ax1.plot(np.array(cumODamage)-np.array(cumADamage), label = log)

	ax1.set(xlabel = 'Games played', ylabel = 'Cumulative net average damage')
	ax1.legend()
	ax1.grid()

	plt.show()

	#Find the first time someone hits 0 stocks
	#Find the both stock 0, return first tuple, first time this happened, and look at the state before it.
	#lastMomentIdx = np.where((aStock==0)&(oStock==0))[0][0]-1
	#print(aStock[lastMomentIdx], oStock[lastMomentIdx])

if '__name__':
	main()