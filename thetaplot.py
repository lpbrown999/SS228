import numpy as np 
import matplotlib
import matplotlib.pyplot as plt

def main():
	thetafileBase = 'thetas/'
	thetaFolder = 'fighter_11_29/'

	thetaName = 'recovery_a1_match202.npy'

	thetas = np.load(thetafileBase+thetaFolder+thetaName)
	thetas = thetas.reshape(63,int(thetas.size/63))

	# for indS,theta in enumerate(thetas):
	# 	ax1.plot(distances, P1[indS,:] , label=str(theta))
	# ax1.grid()
	# ax1.set(xlabel = 'd', ylabel = '$P(c^1 \mid d)$', xlim = (distances[0], distances[-1]), ylim = (0,1))
	# ax1.legend(title='$\\theta$')
	# fig1.tight_layout()
	# fig1.savefig("prob1.png", dpi = dpi)
	



	fig1, ax1 = plt.subplots() 	#Weights on our x position
	# fig2, ax2 = plt.subplots()	#Weights on rel x

	for indTh,theta in enumerate(thetas[0:18]):
		ax1.plot(np.linspace(0,9,10), theta[-10:], label=str(indTh))
		# ax2.plot(np.linspace(0,1,800), theta[312:812], label = str(indTh))
	ax1.legend()
		# ax2.legend()
	plt.show()





if __name__ == "__main__":
	np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
	main()
