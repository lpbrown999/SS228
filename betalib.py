import sys
import select
import pandas as pd
import numpy as np
import scipy.stats

#stateVal should always be concatenation of selfState, oppState
def jumper_animation(stateVal):
	#Define if we are on the ground
	ay = stateVal[1]
	anim = int(stateVal[5])
	onground  = int(stateVal[12] and stateVal[1]<1)

	anim_portion = np.zeros(400) 	  #Approximately 400 animation for now
	if (anim < 400) and (anim >= 0):
		anim_portion[anim] = 1		  #Turn on the theta weight associated with the current animation

	#Just flag of if we are on the ground, and the animation gector
	beta = np.concatenate((np.array([onground]),anim_portion))

	return beta

def jumper_animation_new(stateVal):
	#Agent x,y, animation value
	ax = stateVal[0]
	ay = stateVal[1]
	
	anim = int(stateVal[5])
	onground  = int(stateVal[12] and stateVal[1]<1)

	anim_portion_beta = np.zeros(400) #Approximately for now
	if (anim < 400) and (anim >= 0):
		anim_portion_beta[anim] = 1		  #Turn on the theta weight associated with the current animation

	#x basis functions centered at various locations
	sigma = 10
	x_portion = scipy.stats.norm.pdf(ax,np.array(range(-300,300)),sigma)

	#Facing  - 1 for right, 0 for left
	facingVal = stateVal[4]
	facing_portion = np.zeros(2)
	facing_portion[int(facingVal)] = 1

	#Assemble all basis functions
	beta = np.concatenate( (np.array([0,onground]), anim_portion_beta, x_portion, facing_portion) )
	return beta

def fighter(stateVal):
	#Agent values
	aX 				= stateVal[0]
	aY 				= stateVal[1]
	aPercent		= stateVal[2]
	aStock			= stateVal[3]
	aFacing 		= int(stateVal[4])
	aAnimVal 		= int(stateVal[5])
	aAnimFrame 		= stateVal[6]
	aInvincible 	= stateVal[7]
	aHitLag 		= stateVal[8]
	aHitStun		= stateVal[9]
	aChargeSmash 	= stateVal[10]
	aJumpLeft 		= int(stateVal[11])
	aOnGround 		= stateVal[12]
	aXSpeed 		= stateVal[13]
	aYSpeed 		= stateVal[14]
	aOffStage 		= int(stateVal[15])

	#Opponent values
	oppOffset = 16
	oX 				= stateVal[0+oppOffset]
	oY 				= stateVal[1+oppOffset]
	oPercent		= stateVal[2+oppOffset]
	oStock 			= stateVal[3+oppOffset]
	oFacing 		= int(stateVal[4+oppOffset])
	oAnimVal		= int(stateVal[5+oppOffset])
	oAnimFrame 		= stateVal[6+oppOffset]
	oInvincible 	= stateVal[7+oppOffset]
	oHitLag 		= stateVal[8+oppOffset]
	oHitStun		= stateVal[9+oppOffset]
	oChargeSmash 	= stateVal[10+oppOffset]
	oJumpLeft 		= int(stateVal[11+oppOffset])
	oOnGround 		= stateVal[12+oppOffset]
	oXSpeed 		= stateVal[13+oppOffset]
	oYSpeed 		= stateVal[14+oppOffset]
	oOffStage 		= int(stateVal[15+oppOffset])

	#params
	stageWidth = 85
	stageHeight = 100
	sigmaPos = 3
	sigmaRel = .5

	pctgMax = 200
	sigmaPctg = 10

	#Positional basis functions
	aXBasis = scipy.stats.norm.pdf(aX,np.linspace(-stageWidth-40,stageWidth+40,251),sigmaPos)
	aYBasis = scipy.stats.norm.pdf(aX,np.linspace(-20,stageHeight,61),sigmaPos)

	#Relative positon basis functions -> make very tight so the guy can extrapolate better
	relXBasis = scipy.stats.norm.pdf(aX - oX, np.linspace(-stageWidth*2,stageWidth*2,500)    ,sigmaPos)
	relYBasis = scipy.stats.norm.pdf(aY - oY, np.linspace(-stageHeight-20,stageHeight+20,500),sigmaPos)

	#Facing basis
	facingBasis = np.zeros(4)
	facingBasis[aFacing] = 1
	facingBasis[oFacing+2] = 1

	#Animation basis #Turn on the theta weight associated with the current animation
	animBasis = np.zeros(800)
	if (aAnimVal < 400) and (aAnimVal >= 0):
		animBasis[aAnimVal] = 1		  	
	if (oAnimVal < 400) and (oAnimVal >= 0):
		animBasis[oAnimVal+400] = 1

	#Percentage basis functions
	aPctgBasis = scipy.stats.norm.pdf(aPercent,np.linspace(0,pctgMax,41),sigmaPctg)
	oPctgBasis = scipy.stats.norm.pdf(oPercent,np.linspace(0,pctgMax,41),sigmaPctg)

	#Jumps Left
	jumpLeftBasis = np.zeros(6)
	jumpLeftBasis[aJumpLeft] = 1
	jumpLeftBasis[oJumpLeft+3] = 1

	#Offstage
	offStageBasis = np.zeros(4)
	offStageBasis[aOffStage] = 1
	offStageBasis[oOffStage+2] = 1

	#Assemble all basis functions
	beta = np.concatenate((aXBasis,aYBasis,relXBasis,relYBasis,facingBasis,animBasis,aPctgBasis,oPctgBasis,jumpLeftBasis,offStageBasis))
	return beta

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Remeber to update this dictionary when adding a new beta function #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

betaDict = {"1": jumper_animation,		
			"2": jumper_animation_new,
			"3": fighter}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Remeber to update this dictionary when adding a new beta function #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #