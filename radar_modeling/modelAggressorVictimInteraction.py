# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 14:49:18 2021

@author: Sai Gunaranjan Pelluri
"""


"""

Notes:
    This script models the all the aggressor frequencies that could bias a given victim frequency.
    Below are the algorithmic steps to implement and model the same.

    1. Set the victim frequency/Doppler.
    2. Obtain the baseband frequency fb of the victim @Fs1 rate and the corresponding integer hypothesis m
    3. Eliminate the hypothesis m from the hypothesis Dictionary M
    4. Construct the other frequencies as fb + ({M}-{m})*Fs1. (All these frequencies also alias to the same baseband frequency fb when sampled at Fs1 rate)
    5. Obtain the baseband frequencies(fb_) of the above frequencies (from step 4) when sampled at Fs2 rate. Mathematically, modulo of the above frequencies wrt Fs2.
    6. Construct the true aggressor frequencies as fb_ + M*Fs2. These are all the frequencies(aggressors) that could bias the true victim frequency.

"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

Ts1 = 37e-6
Ts2 = 40e-6
Fs1 = 1/Ts1
Fs2 = 1/Ts2
lightSpeed = 3e8
centerFreq = 76.5e9
lamda = lightSpeed/centerFreq
intHypDict = np.arange(-2,3)
numHyp = len(intHypDict)


numVel = 50 # Number of Velocity points to evaluate
baseBandMaxVel_mps = Fs1/2 * lamda/2
"""
1. Set the victim frequency/Doppler.
2. Obtain the baseband frequency fb of the victim @Fs1 rate and the corresponding integer hypothesis m """
baseBandVel_mps = np.linspace(-baseBandMaxVel_mps, baseBandMaxVel_mps, numVel)
victimVel_mps = baseBandVel_mps[:,None] + intHypDict[None,:]*Fs1*lamda/2
victimVelFlatten_mps = victimVel_mps.T.flatten()

intHypDictMatrix = np.tile(intHypDict[:,None], numHyp)

""" 3. Eliminate the hypothesis m from the hypothesis Dictionary M"""
aggHyp = np.zeros((numHyp-1, numHyp))
for ele in range(numHyp):
    aggHyp[:,ele] = intHypDictMatrix[ele!=np.arange(numHyp),ele]
"""
aggHypS =         [[-1,-2,-2,-2,-2],\
                   [ 0, 0,-1,-1,-1],\
                   [ 1, 1, 1, 0, 0],\
                   [ 2, 2, 2, 2, 1]]
"""

aggHyp = aggHyp.T

allVelatBaseBand = (np.tile(baseBandVel_mps[:,None],numHyp)) # All velocities at baseband

"""4. Construct the other frequencies as fb + ({M}-{m})*Fs1.
(All these frequencies also alias to the same baseband frequency fb when sampled at Fs1 rate)"""
aggressVel_Fs1 =  allVelatBaseBand[:,:,None] + aggHyp[None,:,:]*Fs1*lamda/2
aggressVel_Fs1_flatten = ((np.transpose(aggressVel_Fs1, (2,1,0))).reshape(numHyp-1, numHyp*numVel)).T

"""5. Obtain the baseband frequencies(fb_) of the above frequencies (from step 4) when sampled at Fs2 rate.
Mathematically, modulo of the above frequencies wrt Fs2."""
aggressVelBaseBandFs2 = np.mod(aggressVel_Fs1_flatten, Fs2*lamda/2)
aggressVelBaseBandFs2[aggressVelBaseBandFs2>=((Fs2/2)*(lamda/2))] -=  Fs2*lamda/2 # Is this line required to make the baseband frequencies at Fs2 bi-polar???

""" 6. Construct the true aggressor frequencies as fb_ + M*Fs2.
These are all the frequencies(aggressors) that could bias the true victim frequency."""
aggressVel_Fs2 = aggressVelBaseBandFs2[:,:,None] + intHypDict[None,None, :]*Fs2*lamda/2
aggressVel_Fs2_flatten = aggressVel_Fs2.reshape(-1,(numHyp-1)*numHyp)

victimVelFlattenRepeat_mps = np.tile(victimVelFlatten_mps[:,None],(numHyp-1)*numHyp)


plt.figure(1, figsize=(20,10))
plt.title('Aggressor Velocities for every Victim velocity')
plt.scatter(victimVelFlattenRepeat_mps*3.6,aggressVel_Fs2_flatten*3.6, facecolors='none', edgecolors='r')
plt.xlabel('Victim Speed (kmph)')
plt.ylabel('Aggressor Speed (kmph)')
plt.grid('True')



