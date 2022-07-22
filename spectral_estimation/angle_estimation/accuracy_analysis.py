# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 17:53:54 2022

@author: saiguna
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
numRx = 74
numAngleFFT = 8192 # 8192
mimoArraySpacing = 2e-3 # 2mm
lightSpeed = 3e8
centerFreq = 76.5e9 # GHz
lamda = lightSpeed/centerFreq
Fs_spatial = lamda/mimoArraySpacing
angAxis_deg = np.arcsin(np.arange(-numAngleFFT//2, numAngleFFT//2)*(Fs_spatial/numAngleFFT))*180/np.pi


""" RF parameters """
thermalNoise = -174 # dBm/Hz
noiseFigure = 10 # dB
baseBandgain = 34 #dB
adcSamplingRate = 56.25e6 # 56.25 MHz
adcSamplingTime = 1/adcSamplingRate # s

dBFs_to_dBm = 10
totalNoisePower_dBm = thermalNoise + noiseFigure + baseBandgain + 10*np.log10(adcSamplingRate)
totalNoisePower_dBFs = totalNoisePower_dBm - 10
noiseFloor_perBin = totalNoisePower_dBFs - 10*np.log10(numRx) # dBFs/bin
noisePower_perBin = 10**(noiseFloor_perBin/10)
totalNoisePower = noisePower_perBin*numRx # sigmasquare totalNoisePower
sigma = np.sqrt(totalNoisePower)

numMonteCarloRuns = 1000
binSNRdBArray = np.arange(-10,60,2) #np.arange(18,70,5)
binSNRlinArray = 10**(binSNRdBArray/10)

crlb = 1*1.2/(np.sqrt(2*binSNRlinArray))

numSNRPoints = len(binSNRdBArray)
angErrorArray = np.zeros((numSNRPoints,numMonteCarloRuns))
count = 0

for binSNR in binSNRdBArray:
    signalPowerdBFs = noiseFloor_perBin + binSNR
    signalPower = 10**(signalPowerdBFs/10)
    signalAmplitude = np.sqrt(signalPower)
    signalPhase = np.exp(1j*np.random.uniform(-np.pi, np.pi, numMonteCarloRuns))
    signalphasor = signalAmplitude*signalPhase

    objectAngle_deg = np.zeros((numMonteCarloRuns,)) #np.random.uniform(-50,50, numMonteCarloRuns)
    objectAngle_rad = (objectAngle_deg/360) * (2*np.pi)

    rxSignal = signalphasor[:,None]*np.exp(1j*(2*np.pi/lamda)*mimoArraySpacing*np.sin(objectAngle_rad[:,None])*np.arange(numRx)[None,:]) # [numAngMonteCarlo, numRx]
    noise = (sigma/np.sqrt(2))*np.random.randn(numRx*numMonteCarloRuns) + 1j*(sigma/np.sqrt(2))*np.random.randn(numRx*numMonteCarloRuns)
    noise = noise.reshape(numMonteCarloRuns,numRx) # [numMonteCarlo, numRx]
    signal = rxSignal + noise

    signalFFT = np.fft.fft(signal,axis=1,n=numAngleFFT)/numRx
    signalFFTshift = np.fft.fftshift(signalFFT,axes=(1,))
    magSpec = np.abs(signalFFTshift)**2
    angInd = np.argmax(magSpec,axis=1)
    estAngDeg = angAxis_deg[angInd]
    angDegError = objectAngle_deg - estAngDeg
    angErrorArray[count,:] = angDegError
    count += 1

mseAngle = np.mean(np.abs(angErrorArray)**2,axis=1) #np.std(angErrorArray,axis=1)#np.mean(np.abs(angErrorArray)**2,axis=1)
varianceAngEst = mseAngle #mseAngle**2
stdAngEst = np.sqrt(varianceAngEst) #mseAngle #np.sqrt(varianceAngEst)


plt.figure(1,figsize=(20,10),dpi=200)
plt.title('Angle error std(dB) vs SNR (dB)')
# plt.plot(binSNRdBArray,stdAngEst, '-o')
# plt.plot(binSNRlinArray,stdAngEst, '-o')
# plt.plot(binSNRlinArray,10*np.log10(stdAngEst), '-o')
plt.plot(binSNRdBArray,20*np.log10(stdAngEst), '-o')
plt.plot(binSNRdBArray, 20*np.log10(crlb))
# plt.xscale('log')
plt.xlabel('SNR')
plt.ylabel('std (dB)')
plt.grid(True)


plt.figure(2,figsize=(20,10),dpi=200)
plt.title('Angle error std(deg)  vs SNR (dB)')
# plt.plot(binSNRdBArray,stdAngEst, '-o')
# plt.plot(binSNRlinArray,stdAngEst, '-o')
# plt.plot(binSNRlinArray,10*np.log10(stdAngEst), '-o')
plt.plot(binSNRdBArray,stdAngEst, '-o')
# plt.xscale('log')
plt.xlabel('SNR')
plt.ylabel('std (deg)')
plt.grid(True)
