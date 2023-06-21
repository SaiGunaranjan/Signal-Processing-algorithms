# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 17:53:54 2022

@author: saiguna
"""

""" Borrowed most of the script from angle_estimation//accuracy_analysis.py script
and commented out the FFT part of the analysis"""


import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..//')
from spectral_estimation_lib import music_snapshots

plt.close('all')
numRx = 6#74
numPointsAngleFFT = np.array([256])
lenNumAngFFTInstances = len(numPointsAngleFFT)
mimoArraySpacing = 2e-3 # 2mm
lightSpeed = 3e8
centerFreq = 76.5e9 # GHz
lamda = lightSpeed/centerFreq
Fs_spatial = lamda/mimoArraySpacing
digital_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/(numPointsAngleFFT))



""" RF parameters """
thermalNoise = -174 # dBm/Hz
noiseFigure = 10 # dB
baseBandgain = 34 #dB
adcSamplingRate = 56.25e6 # MHz
adcSamplingTime = 1/adcSamplingRate # s

dBFs_to_dBm = 10
totalNoisePower_dBm = thermalNoise + noiseFigure + baseBandgain + 10*np.log10(adcSamplingRate)
totalNoisePower_dBFs = totalNoisePower_dBm - 10
noiseFloor_perBin = totalNoisePower_dBFs - 10*np.log10(numRx) # dBFs/bin
noisePower_perBin = 10**(noiseFloor_perBin/10)
totalNoisePower = noisePower_perBin*numRx # sigmasquare totalNoisePower
sigma = np.sqrt(totalNoisePower)

numMonteCarloRuns = 1000
numSnapshots = 15#50
num_sources = 1
binSNRdBArray = np.arange(-10,44,2)
binSNRlinArray = 10**(binSNRdBArray/10)

dBCLevelBeamWidth = 10

""" The below formula is the Cramer Rao Lower bound limit for accuracy of an estimator.
It is given by resolution/sqrt(2*binSNRLinearScale)"""
crlb = 1.2/(np.sqrt(2*binSNRlinArray))

numSNRPoints = len(binSNRdBArray)
angErrorArray = np.zeros((numSNRPoints,numMonteCarloRuns))
beamWidthArray = np.zeros((numSNRPoints,numMonteCarloRuns))
count = 0

for binSNR in binSNRdBArray:
    signalPowerdBFs = noiseFloor_perBin + binSNR
    signalPower = 10**(signalPowerdBFs/10)
    signalAmplitude = np.sqrt(signalPower)
    signalPhase = np.exp(1j*np.random.uniform(-np.pi, np.pi, numMonteCarloRuns*numSnapshots))
    signalPhase = signalPhase.reshape(numMonteCarloRuns, numSnapshots)
    signalphasor = signalAmplitude*signalPhase

    objectAngle_deg = np.random.uniform(-50,50, numMonteCarloRuns)#np.zeros((numMonteCarloRuns,))
    objectAngle_deg = objectAngle_deg.reshape(numMonteCarloRuns)
    objectAngle_rad = (objectAngle_deg/360) * (2*np.pi)

    rxSignal = signalphasor[:,None,:]*np.exp(1j*(2*np.pi/lamda)*mimoArraySpacing*np.sin(objectAngle_rad[:,None,None])*np.arange(numRx)[None,:,None]) # [numAngMonteCarlo, numRx]
    noise = (sigma/np.sqrt(2))*np.random.randn(numRx*numMonteCarloRuns*numSnapshots) + 1j*(sigma/np.sqrt(2))*np.random.randn(numRx*numMonteCarloRuns*numSnapshots)
    noise = noise.reshape(numMonteCarloRuns,numRx,numSnapshots) # [numMonteCarlo, numRx, numSnapshots]
    signal = rxSignal + noise

    # signalFFT = np.fft.fft(signal,axis=1,n=numPointsAngleFFT[0])/numRx
    # signalFFTshift = np.fft.fftshift(signalFFT,axes=(1,))
    # magSpec = np.abs(signalFFTshift)**2
    # magSpec = magSpec[:,:,0] # Sampling the zero snapshot for FFT

    for monteCarloIter in range(numMonteCarloRuns):
        received_signal = signal[monteCarloIter,:,:]
        pseudo_spectrum = music_snapshots(received_signal, num_sources, numRx, digital_freq_grid)
        pseudo_spectrum = pseudo_spectrum/np.amax(pseudo_spectrum)
        pseudo_spectrumdB = 10*np.log10(pseudo_spectrum)
        angInd = np.argmax(pseudo_spectrum)
        angAxis_deg = np.arcsin(np.arange(-numPointsAngleFFT//2, numPointsAngleFFT//2)*(Fs_spatial/numPointsAngleFFT))*180/np.pi
        estAngDeg = angAxis_deg[angInd]
        angDegError = objectAngle_deg[monteCarloIter] - estAngDeg
        angErrorArray[count,monteCarloIter] = angDegError

        binWidthArr = np.argsort(np.abs(pseudo_spectrumdB + dBCLevelBeamWidth))[0:2]
        binWidth = np.abs(np.diff(binWidthArr))
        beamWidth = np.arcsin(binWidth*(Fs_spatial/numPointsAngleFFT))*180/np.pi
        beamWidthArray[count,monteCarloIter] = beamWidth



    # angInd = np.argmax(magSpec,axis=1)
    # angAxis_deg = np.arcsin(np.arange(-numPointsAngleFFT//2, numPointsAngleFFT//2)*(Fs_spatial/numPointsAngleFFT))*180/np.pi
    # estAngDeg = angAxis_deg[angInd]
    # angDegError = objectAngle_deg - estAngDeg
    # angErrorArray[count,:] = angDegError


    count += 1

""" Both the below methods of computing the variance and std are similar,
but better to go with the second method that I'm using currently """

# mseAngle = np.mean(np.abs(angErrorArray)**2,axis=1)
# varianceAngEst = mseAngle
# stdAngEst = np.sqrt(varianceAngEst)

stdAngEst = np.std(angErrorArray,axis=1)
varianceAngEst = stdAngEst**2

binRes = np.arcsin(Fs_spatial/numPointsAngleFFT)*180/np.pi
variancebinRes = binRes**2 / 12
binResdB = 10*np.log10(variancebinRes)
sigmabinRes = np.sqrt(variancebinRes)

meanbeamWidth = np.mean(beamWidthArray,axis=1)

legendArray = [str(x) + " point MUSIC" for x in numPointsAngleFFT]

plt.figure(1,figsize=(20,10),dpi=200)
plt.title('Angle error std(dB) vs SNR (dB) with {} point MUSIC'.format(numPointsAngleFFT[0]))
plt.plot(binSNRdBArray,20*np.log10(stdAngEst), '-o')
# plt.plot(binSNRdBArray, 20*np.log10(crlb), color='k')
plt.axhline(binResdB,color='black',ls='dashed',label='bin quant. sigma (dB)')
plt.xlabel('SNR (dB)')
plt.ylabel('std (dB)')
plt.grid(True)
plt.ylim([-40,40])
plt.legend()
# legendArray.append('CRLB limit')
# plt.legend(legendArray)


plt.figure(2,figsize=(20,10),dpi=200)
plt.title('Angle error std(deg)  vs SNR (dB)')
plt.plot(binSNRdBArray,stdAngEst, '-o')
# plt.plot(binSNRdBArray, crlb, color='k')
plt.axhline(sigmabinRes,color='black',ls='dashed',label='bin quant. sigma (deg)')
plt.xlabel('SNR (dB)')
plt.ylabel('std (deg)')
plt.grid(True)
plt.legend()
# legendArray.append('CRLB limit')
# plt.legend(legendArray)


plt.figure(3,figsize=(20,10),dpi=200)
plt.title('{} dB BW (deg)  vs SNR (dB)'.format(dBCLevelBeamWidth))
plt.plot(binSNRdBArray,meanbeamWidth, '-o')
plt.xlabel('SNR (dB)')
plt.ylabel('BW (deg)')
plt.grid(True)

