import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt

def main():
	logs = ['logs/Agent1Logs/noPriorLam50Lvl1.csv',
			'logs/Agent1logs/lvl1Lam50PriorLam200Lvl1.csv',
			'logs/Agent1Logs/nightMondayLVL3.csv']
	logLabels = ['$\lambda = 50$, No prior, Lvl1 opponent',
				'$\lambda = 200$, Lvl1 prior, Lvl1 opponent',
				'$\lambda = 200$, Lvl1 prior, Lvl3 opponent']
	gameLims = [85,121,160]
	N = 15

	gameShift = 0
	fig1, ax1 = plt.subplots()		#Plots for winrate
	fig2, ax2 = plt.subplots()		#Plots for damage
	fig3, ax3 = plt.subplots()		#Plots for end of game stocks	

	for indL,log in enumerate(logs):
		df = pd.read_csv(log, header = None)
		data = df.values
		
		aStock = data[:,3]
		oStock = data[:,3+16]
		aDamage = data[:,2]
		oDamage = data[:,2+16]

		winLoss = []				#1 for win, 0 for loss
		cumWinPctg = []				#Cumulative win percentage - mean up to current game

		agentTotalDamage = []		#Damage taken in each game
		opponentTotalDamage = []	#Damage dealt in each game
		cumADamage = []				#Mean up to current game
		cumODamage = []

		agentEndGameStocks = []		#Num stocks left for agent at end of each game
		opponentEndGameStocks = []	#Num stocks left for opponent at end of each game

		gameCounter = 0
		while True:

			#Find subset of data that is the "game"
			startGameIdx = np.where(aStock == 4)[0]
			if startGameIdx.size == 0:
				break
			else:
				startGameIdx = startGameIdx[0]

			#Find end of game or break if no more games
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
			
			#Determine win or loss, track stocks.
			endGameIdx = min(aidx0,oidx0)
			if aStock[endGameIdx] == 0:
				winLoss.append(0)
			else:
				winLoss.append(1)
			
			cumWinPctg.append(sum(winLoss)/len(winLoss))

			#Damage, Stock subsets fo this game
			aStockSubset = aStock[startGameIdx:endGameIdx+1]
			oStockSubset = oStock[startGameIdx:endGameIdx+1]
			aDamageSubset = aDamage[startGameIdx:endGameIdx+1]
			oDamageSubset = oDamage[startGameIdx:endGameIdx+1]

			agentEndGameStocks.append(aStockSubset[-1])
			opponentEndGameStocks.append(oStockSubset[-1])

			#Agent total damage
			agentTotalDamage.append(0)
			for stock in range(3,-1,-1):
				agentNewLifeIdx = np.where(aStockSubset==stock)[0]
				
				if agentNewLifeIdx.size == 0:						#No more deaths
					agentTotalDamage[-1] += aDamageSubset[-1]
					break
				
				agentTotalDamage[-1] += aDamageSubset[agentNewLifeIdx[0]-1]
			cumADamage.append(sum(agentTotalDamage)/len(agentTotalDamage))
			
			#Opponent total damage
			opponentTotalDamage.append(0)
			for stock in range(3,-1,-1):
				oppNewLifeIdx = np.where(oStockSubset==stock)[0]
				
				if oppNewLifeIdx.size == 0:						#No more deaths
					opponentTotalDamage[-1] += oDamageSubset[-1]
					break

				opponentTotalDamage[-1] += oDamageSubset[oppNewLifeIdx[0]-1]
			cumODamage.append(sum(opponentTotalDamage)/len(opponentTotalDamage))

			#Remove all data before this recorded entry. 
			aStock = aStock[endGameIdx+1:]
			oStock = oStock[endGameIdx+1:]
			aDamage = aDamage[endGameIdx+1:]
			oDamage = oDamage[endGameIdx+1:]

			#Find start of next match, shift forward
			a4idx = np.where((aStock==4)&(oStock==4))[0]
			if (a4idx.size == 0):
				break										
			a4idx = a4idx[0]
			aStock = aStock[a4idx:]
			oStock = oStock[a4idx:]
			aDamage = aDamage[a4idx:]
			oDamage = oDamage[a4idx:]

			gameCounter += 1
			if gameCounter > gameLims[indL]:
				break

		## cumWinPctg
		#ax1.plot( np.array(range(0,len(cumWinPctg))), cumWinPctg, label = log)
		runningMeanWinPctg = running_mean(winLoss, N)
		ax1.plot(np.linspace(N,np.array(winLoss).size,runningMeanWinPctg.size), runningMeanWinPctg, label = logLabels[indL])
	
	ax1.set(xlabel = 'Games Played', ylabel = 'Win Percentage',
		    xlim =(0,max(gameLims)), ylim = (0,1.05) )
	ax1.legend()
	ax1.grid()

	plt.show()

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return ((cumsum[N:] - cumsum[:-N]) / float(N))

if '__name__':
	main()