# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 16:31:01 2022

@author: saiguna
"""

""" This function synthesizes the virtual MIMO phasor for the Steradian modules SRIR144 and SRIR256 which have
12Tx x 12Rx and 16Tx x 16Rx respectively. SRIRR144 is a 3 IC platform with each IC having 4 Txs and 4 Rxs.
Similarly, the SRIR256  is a 4 IC platform. However, the SRIR256 has only 13 usable Txs which effectively
synthesizes a 13Tx x 16 Rx = 208 virtual MIMO elements.
Also, the number of Az ULA elements for the 144 platform is 48 while the number of Az ULA elements
for the 256 platform is 74.

The phasor is computed as exp(1j*2pi/lamda [sin(theta)cos(phi) sin(theta)sin(phi) cos(theta)] [x y 0].T)
where theta is the angle in the plane of phi. x,y,z are the positions of each virtual MIMO channel. Since, all the
antennas are on the plane of the RADAR board, the z co-ordinate of the Antennas is always 0
"""


import numpy as np


def mimoPhasorSynth(platform, lamda, objectAzAngle_rad, objectElAngle_rad):

    path = 'antenna_cordinates\\' + platform + '\\antenna\\'

    if (platform == 'SRIR144'):
        numTx = 12
        numRx = 12
        txSeq = np.arange(numTx)
        ulaInd = np.array([91, 79, 67, 55, 90, 78, 66, 54, 89, 77, 65, 53, 88, 76, 64, 52, 43,
                               31, 19,  7, 42, 30, 18,  6, 41, 29, 17,  5, 40, 28, 16,  4, 39, 27,
                               15,  3, 38, 26, 14,  2, 37, 25, 13,  1, 36, 24, 12,  0]) # 48 element Az ULA


    elif (platform == 'SRIR256'):

        numTx = 13
        numRx = 16
        txSeq = np.array([15,12,11,9,8,7,6,5,4,3,2,1,0])
        ulaInd = np.array([ 47,  46,  45,  44,  63,  62,  61,  60,  79,  78,  77,  76,  95,
                94,  93,  92, 111, 110, 109, 108, 127, 126, 125, 124, 143, 142,
               141, 140, 159, 158, 157, 156, 175, 174, 173, 172, 191, 190, 189,
               188, 207, 206, 205, 204,  85,  84, 103, 102, 101, 100, 119, 118,
               117, 116, 135, 134, 133, 132, 151, 150, 149, 148, 167, 166, 165,
               164, 183, 182, 181, 180, 199, 198, 197, 196]) # 74 element Az ULA


    physicalTxCordinates = np.loadtxt(path+'LTXMA.txt')
    physicalRxCordinates = np.loadtxt(path+'LRXMA.txt')

    SeqBasedTxCordinates = physicalTxCordinates[txSeq,:]
    SeqBasedRxCordinates = np.copy(physicalRxCordinates)

    virtualArrayCordinates = SeqBasedTxCordinates[:,:,None] + SeqBasedRxCordinates.T[None,:,:] # [numTx, 3, numRx]
    virtualArrayCordinates = np.transpose(virtualArrayCordinates, (1,0,2)).reshape(3,numTx*numRx) # [3, numTx x numRx]

    azComp = np.sin(objectAzAngle_rad)*np.cos(objectElAngle_rad)
    elComp = np.sin(objectAzAngle_rad)*np.sin(objectElAngle_rad)
    radialComp = np.cos(objectAzAngle_rad)

    objUnitVector = np.vstack((azComp,elComp,radialComp)).T # [numObj, 3]

    beta = 2*np.pi/lamda
    mimoPhasor = np.exp(1j * beta * (objUnitVector @ virtualArrayCordinates)) # [numObj, numTx x numRx]

    # ulaCoeff = mimoPhasor[:,ulaInd] # for debug and cheking if we are obtaining a linear phase across the ULA elements

    mimoPhasor_txrx = mimoPhasor.reshape(-1, numTx, numRx) # [numObj, numTx, numRx]

    return mimoPhasor, mimoPhasor_txrx, ulaInd


