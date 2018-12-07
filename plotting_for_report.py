import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt

def main():
	logs = ['logs/Agent1Logs/noPriorLam50Lvl1.csv',
			'logs/Agent1logs/lvl1Lam50PriorLam200Lvl1.csv',
			'logs/Agent1Logs/nightMondayLVL3.csv',
			'logs/Agent1logs/LVL4.csv']
	logLabels = ['$\lambda = 50$, No prior, Lvl1 opponent, 85 games',
				'$\lambda = 200$, Lvl1 prior, Lvl1 opponent, 120 games',
				'$\lambda = 200$, Lvl1 prior, Lvl3 opponent, 160 games',
				'$\lambda = 200$, Lvl3 prior, Lvl4 opponent, 245 games']
	gameLims = [85,121,160, 245]
	N = 25

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

		agentDamageTaken = []		#Damage taken in each game
		opponentDamageTaken = []	#Damage dealt in each game
		cumADamage = []				#Mean up to current game
		cumODamage = []

		agentEndgameStocks = []		#Num stocks left for agent at end of each game
		opponentEndgameStocks = []	#Num stocks left for opponent at end of each game
		cumAEndStocks = []
		cumOEndStocks = []

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

			agentEndgameStocks.append(aStockSubset[-1])
			opponentEndgameStocks.append(oStockSubset[-1])

			cumAEndStocks.append(sum(agentEndgameStocks)/len(agentEndgameStocks))
			cumOEndStocks.append(sum(opponentEndgameStocks)/len(opponentEndgameStocks))

			#Agent total damage
			agentDamageTaken.append(0)
			for stock in range(3,-1,-1):
				agentNewLifeIdx = np.where(aStockSubset==stock)[0]
				
				if agentNewLifeIdx.size == 0:						#No more deaths
					agentDamageTaken[-1] += aDamageSubset[-1]
					break
				
				agentDamageTaken[-1] += aDamageSubset[agentNewLifeIdx[0]-1]
			cumADamage.append(sum(agentDamageTaken)/len(agentDamageTaken))
			
			#Opponent total damage
			opponentDamageTaken.append(0)
			for stock in range(3,-1,-1):
				oppNewLifeIdx = np.where(oStockSubset==stock)[0]
				
				if oppNewLifeIdx.size == 0:						#No more deaths
					opponentDamageTaken[-1] += oDamageSubset[-1]
					break

				opponentDamageTaken[-1] += oDamageSubset[oppNewLifeIdx[0]-1]
			cumODamage.append(sum(opponentDamageTaken)/len(opponentDamageTaken))


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

		#Setup plot vecs
		runWinPctg = running_mean(winLoss, N)
		
		runAgentDamageTaken = running_mean(agentDamageTaken, N)
		runOpponentDamageTaken = running_mean(opponentDamageTaken, N)
		runDiffDamageTaken = runOpponentDamageTaken-runAgentDamageTaken

		runAgentEndgameStocks = running_mean(agentEndgameStocks,N)
		runOpponentEndgameStocks = running_mean(opponentEndgameStocks,N)
		runDiffStocks = runAgentEndgameStocks-runOpponentEndgameStocks

		gameArray = np.linspace(N,np.array(winLoss).size,runWinPctg.size)

		cumWinPctg
		cumDiffDamage = np.array(cumODamage) -  np.array(cumADamage)
		cumDiffStocks = np.array(cumAEndStocks) - np.array(cumOEndStocks)

		#Plot
		#Running mean plots -> Mean over last N
		# ax1.plot(gameArray, runWinPctg, label = logLabels[indL])
		# ax2.plot(gameArray, runDiffDamageTaken, label = logLabels[indL])
		# ax3.plot(gameArray, runDiffStocks, label=logLabels[indL])
		ax1.plot(np.linspace(0,1,len(runWinPctg)) , runWinPctg, 		 label = logLabels[indL])
		ax2.plot(np.linspace(0,1,len(runWinPctg)) , runDiffDamageTaken, label = logLabels[indL])
		ax3.plot(np.linspace(0,1,len(runWinPctg)) , runDiffStocks, 	 label=logLabels[indL])


		#Cumulative mean plots -> Mean over all games up to point
		# ax1.plot(gameArray, cumWinPctg[N-1:], label = logLabels[indL])
		# ax2.plot(gameArray, cumDiffDamage[N-1:], label = logLabels[indL])
		# ax3.plot(gameArray, cumDiffStocks[N-1:], label = logLabels[indL])

		
	ax1.set(xlabel = 'Percent games completed', ylabel = 'Win percentage', xlim =(0,1), ylim = (0,1.05) )
	ax1.legend(fontsize ='x-small')
	ax1.grid()
	fig1.savefig("winpctg.png", dpi = 300)

	ax2.set(xlabel = 'Percent games completed', ylabel = 'Difference in total damage dealt', xlim =(0,1))
	ax2.legend(fontsize ='x-small')
	ax2.grid()
	fig2.savefig("damage.png", dpi = 300)

	ax3.set(xlabel = 'Percent games completed', ylabel = 'Stock difference at end of game', xlim =(0,1))
	ax3.legend(fontsize ='x-small')
	ax3.grid()
	fig3.savefig("stocks.png", dpi = 300)

	plt.show()

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return ((cumsum[N:] - cumsum[:-N]) / float(N))

if '__name__':
	main()