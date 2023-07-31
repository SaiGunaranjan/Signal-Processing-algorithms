# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:04:02 2023

@author: Sai Gunaranjan
"""


"""
Array 1 (L shaped array): Tx separated by lamda vertically and Rxs separated by lamda/2 horizontally

      Tx1

      Tx2

      Tx3

      Tx4

      Tx5

      Tx6


      Rx1 Rx2 Rx3 Rx4 Rx5 Rx6



Array 2: Tx2, Tx3, Tx4 separated by 3lamda/2 horizontally.
Tx1 and Tx2 separated by 2lamda in vertical.
Rx2, Rx3, Rx4 separated by lamda/2.
Rx1 and Rx2 separated by lamda.



      Tx2   Tx3   Tx4



      Tx1

      Rx2 Rx3 Rx4

      Rx1

So the virtual array for array 2, in essence, the azimuth ULA has (3Tx x 3Rx)9 elements separated by lamda/2 and
elevation ULA has 4 elements separated by lamda. So the virtual array for array2 looks as below:

    * * * * * * * * *

    *

    *

    *


"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../..")
sys.path.append("..")
from spectral_estimation.spectral_estimation_lib import music_snapshots
from scipy.signal import argrelextrema


np.random.seed(5)
plt.close('all')

def mimoPhasorSynth(lamda, objectAzAngle_rad, objectElAngle_rad):

    """ L_shaped_array """
    # numTx and numRx available as global variables

    txSeq = np.arange(numTx)
    ulaInd = np.arange(numRx)

    tx_Yoffset = 0#10e-3
    tx_ycordinates = tx_Yoffset + txSpacing*np.arange(numTx)

    rx_Xoffset = 0#20e-3
    rx_xcordinates = rx_Xoffset + rxSpacing*np.arange(numRx)

    physicalTxCordinates = np.zeros((numTx,3),dtype=np.float32) # [x,y,z]
    physicalTxCordinates[:,1] = np.flipud(tx_ycordinates)

    physicalRxCordinates = np.zeros((numRx,3),dtype=np.float32)
    physicalRxCordinates[:,0] = np.flipud(rx_xcordinates)


    SeqBasedTxCordinates = physicalTxCordinates[txSeq,:]
    SeqBasedRxCordinates = np.copy(physicalRxCordinates)

    virtualArrayCordinates = SeqBasedTxCordinates[:,:,None] + SeqBasedRxCordinates.T[None,:,:] # [numTx, 3, numRx]
    virtualArrayCordinates = np.transpose(virtualArrayCordinates, (1,0,2)).reshape(3,numTx*numRx) # [3, numTx x numRx]

    # for debug
    # plt.scatter(physicalRxCordinates[:,0],physicalRxCordinates[:,1])
    # plt.scatter(physicalTxCordinates[:,0],physicalTxCordinates[:,1])
    # plt.gca().set_aspect('equal')


    azComp = np.sin(objectAzAngle_rad)
    elComp = np.sin(objectElAngle_rad)
    radialComp = np.cos(objectAzAngle_rad)

    # azComp = np.sin(objectAzAngle_rad)*np.cos(objectElAngle_rad)
    # elComp = np.sin(objectAzAngle_rad)*np.sin(objectElAngle_rad)
    # radialComp = np.cos(objectAzAngle_rad)

    objUnitVector = np.vstack((azComp,elComp,radialComp)).T # [numObj, 3]

    beta = 2*np.pi/lamda
    mimoPhasor = np.exp(1j * beta * (objUnitVector @ virtualArrayCordinates)) # [numObj, numTx x numRx]

    mimoPhasor_txrx = mimoPhasor.reshape(-1, numTx, numRx) # [numObj, numTx, numRx]

    return mimoPhasor, mimoPhasor_txrx, ulaInd




platform = 'L_shaped_array'
lightSpeed = 3e8
centerFreq = 76.5e9
lamda = lightSpeed/centerFreq
numTx = 6
numRx = 6
rxSpacing = lamda/2
fsRx = lamda/rxSpacing
txSpacing = lamda #lamda/2
fsTx = lamda/txSpacing
num_sources = 3

rxAngRes = np.arcsin(fsRx/numRx)*180/np.pi
txAngRes = np.arcsin(fsTx/numTx)*180/np.pi

rxMaxAng = np.arcsin(fsRx/2)*180/np.pi
txMaxAng = np.arcsin(fsTx/2)*180/np.pi

print('Rx Ang Res = {0:.2f} deg, Tx Ang Res = {1:.2f} deg'.format(rxAngRes,txAngRes))
print('Rx Ang Max = {0:.2f} deg, Tx Ang Max = {1:.2f} deg'.format(rxMaxAng,txMaxAng))

spectrumGridOSRFact = 43#To obtain about 256 sample MUSIC
digital_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/(spectrumGridOSRFact*numTx))
numPointMUSIC = len(digital_freq_grid)

angleGridRx = np.arcsin(((digital_freq_grid/(2*np.pi))*fsRx))*180/np.pi
AngbinResRx = np.arcsin(fsRx/numPointMUSIC)*180/np.pi
angleGridTx = np.arcsin(((digital_freq_grid/(2*np.pi))*fsTx))*180/np.pi
AngbinResTx = np.arcsin(fsTx/numPointMUSIC)*180/np.pi

objectAzAngle_deg = np.array([-5,5,10])
objectElAngle_deg = np.array([-10,-5, 2])

objectAzAngle_rad = (objectAzAngle_deg/360) * (2*np.pi)
objectElAngle_rad = (objectElAngle_deg/360) * (2*np.pi)
actualAzElAnglePairs = np.hstack((objectAzAngle_deg[:,None],objectElAngle_deg[:,None]))

_, mimoPhasor_txrx, _ = mimoPhasorSynth(lamda, objectAzAngle_rad, objectElAngle_rad)

ver_ula_signal = np.conj(mimoPhasor_txrx[:,:,0].T)
hor_ula_signal = np.conj(mimoPhasor_txrx[:,0,:].T)

numSnapshots = 100
snr = 50
snrdelta = 0#3 # This indicates by how much dB is the second target below the 1st target and so on
object_snr = np.array([snr,snr-snrdelta,snr-2*snrdelta])
""" Noise parameters"""
noiseFloordB = -100
noise_power_db = noiseFloordB + 10*np.log10(numTx)
noise_variance = 10**(noise_power_db/10)
noise_sigma = np.sqrt(noise_variance)
weights = np.sqrt(10**((noiseFloordB + object_snr)/10))

signal_phases = np.exp(1j*np.random.uniform(low=-np.pi, high = np.pi, size = numSnapshots*num_sources))
signal_phases = signal_phases.reshape(num_sources,numSnapshots)
complex_signal_amplitudes = weights[:,None]*signal_phases
ver_signal = ver_ula_signal @ complex_signal_amplitudes
hor_signal = hor_ula_signal @ complex_signal_amplitudes

wgn_noise_tx = (noise_sigma/np.sqrt(2))*np.random.randn(numTx * numSnapshots) + 1j*(noise_sigma/np.sqrt(2))*np.random.randn(numTx * numSnapshots)
ver_signal = ver_signal + wgn_noise_tx.reshape(-1,numSnapshots)

wgn_noise_rx = (noise_sigma/np.sqrt(2))*np.random.randn(numRx * numSnapshots) + 1j*(noise_sigma/np.sqrt(2))*np.random.randn(numRx * numSnapshots)
hor_signal = hor_signal + wgn_noise_rx.reshape(-1,numSnapshots)

pseudo_spectrum_rx = music_snapshots(hor_signal, num_sources, numRx, digital_freq_grid)
pseudo_spectrum_rx = pseudo_spectrum_rx/np.amax(pseudo_spectrum_rx)
pseudo_spectrum_rxdB = 10*np.log10(pseudo_spectrum_rx)

pseudo_spectrum_tx = music_snapshots(ver_signal, num_sources, numTx, digital_freq_grid)
pseudo_spectrum_tx = pseudo_spectrum_tx/np.amax(pseudo_spectrum_tx)
pseudo_spectrum_txdB = 10*np.log10(pseudo_spectrum_tx)

localMaxInd = argrelextrema(pseudo_spectrum_rxdB,np.greater,axis=0,order=1)[0]
peakInd = np.argsort(pseudo_spectrum_rxdB[localMaxInd])[-num_sources::]
localMaxPeaks = localMaxInd[peakInd]
azDigFreq = digital_freq_grid[localMaxPeaks]
estAzAngles = angleGridRx[localMaxPeaks]

localMaxInd = argrelextrema(pseudo_spectrum_txdB,np.greater,axis=0,order=1)[0]
peakInd = np.argsort(pseudo_spectrum_txdB[localMaxInd])[-num_sources::]
localMaxPeaks = localMaxInd[peakInd]
elDigFreq = digital_freq_grid[localMaxPeaks]
estElAngles = angleGridTx[localMaxPeaks]

actualAzElAnglePairs = actualAzElAnglePairs[np.argsort(actualAzElAnglePairs[:,0]),:]

print('\n True Az/El angle pairs = \n', actualAzElAnglePairs)


plt.figure(1,figsize=(20,10),dpi=200)
plt.subplot(1,2,1)
plt.title('Num Rx samples = ' + str(numRx))
plt.plot(angleGridRx, pseudo_spectrum_rxdB, label='MUSIC snapshots = {}'.format(numSnapshots))
plt.vlines(objectAzAngle_deg,-60,5, alpha=1,color='black',ls='dashed',label = 'Ground truth')
plt.xlabel('Angle(deg)')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.title('Num Tx samples = ' + str(numTx))
plt.plot(angleGridTx, pseudo_spectrum_txdB, label='MUSIC snapshots = {}'.format(numSnapshots))
plt.vlines(objectElAngle_deg,-60,5, alpha=1,color='black',ls='dashed',label = 'Ground truth')
plt.xlabel('Angle(deg)')
plt.legend()
plt.grid(True)
