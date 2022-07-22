# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 17:53:54 2022

@author: saiguna
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../..")
sys.path.append("..")
from function_utilities.libraryFunctions import computeTwoLargestLocalMaxima
from spectral_estimation_lib import iaa_recursive
import time


tstart = time.time()
plt.close('all')

numRx = 74
numAngleFFT = 2048 # 8192
mimoArraySpacing = 2e-3 # 2mm
lightSpeed = 3e8
centerFreq = 76.5e9 # GHz
lamda = lightSpeed/centerFreq
Fs_spatial = lamda/mimoArraySpacing
angAxis_deg = np.arcsin(np.arange(-numAngleFFT//2, numAngleFFT//2)*(Fs_spatial/numAngleFFT))*180/np.pi
digital_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/(numAngleFFT))


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

numMonteCarloRuns = 100#1000
binSNRdBArray = np.array([60]) #np.arange(18,70,5)
binSNRlinArray = 10**(binSNRdBArray/10)
numSNRPoints = len(binSNRdBArray)

numTargets = 2
angSepDeg = np.arange(0.4,4,0.1)
numAngSepPoints = len(angSepDeg)
target1AngDeg = np.zeros((numAngSepPoints))
target2AngDeg = target1AngDeg + angSepDeg
objectAngle_deg = np.vstack((target1AngDeg,target2AngDeg))
objectAngle_deg = objectAngle_deg[:,:,None] * np.ones(numMonteCarloRuns)[None,None,:]
objectAngle_rad = (objectAngle_deg/360) * (2*np.pi)

signalPhase = np.exp(1j*np.random.uniform(-np.pi, np.pi, numMonteCarloRuns*numTargets*numAngSepPoints))
signalPhase = signalPhase.reshape(numTargets,numAngSepPoints,numMonteCarloRuns)

estAngSepArrFFT = np.zeros((numSNRPoints,numAngSepPoints, numMonteCarloRuns))
estAngSepArrFFTWin = np.zeros((numSNRPoints,numAngSepPoints, numMonteCarloRuns))
estAngSepArrIAA = np.zeros((numSNRPoints,numAngSepPoints, numMonteCarloRuns))

numIAAiterations = 10
count = 0

for binSNR in binSNRdBArray:
    signalPowerdBFs = noiseFloor_perBin + binSNR
    signalPower = 10**(signalPowerdBFs/10)
    signalAmplitude = np.sqrt(signalPower)
    signalphasor = signalAmplitude*signalPhase

    rxSignal = signalphasor[:,:,:,None]*np.exp(1j*(2*np.pi/lamda)*mimoArraySpacing*np.sin(objectAngle_rad[:,:,:,None])*np.arange(numRx)[None,None,None,:]) # [2targets, numbinSep, numAngMonteCarlo, numRx]
    rxSignal = np.sum(rxSignal,axis=0) # [numbinSep, numAngMonteCarlo, numRx]
    noise = (sigma/np.sqrt(2))*np.random.randn(numRx*numMonteCarloRuns*numAngSepPoints) + 1j*(sigma/np.sqrt(2))*np.random.randn(numRx*numMonteCarloRuns*numAngSepPoints)
    noise = noise.reshape(numAngSepPoints,numMonteCarloRuns,numRx) # [numbinSep, numMonteCarlo, numRx]

    signal = rxSignal + noise # [numbinSep, numMonteCarlo, numRx]

    """ UnWindowed FFT"""
    signalFFT = np.fft.fft(signal,axis=2,n=numAngleFFT)/numRx # [numbinSep, numMonteCarlo, numRxFFT]
    signalFFTshift = np.fft.fftshift(signalFFT,axes=(2,)) # [numbinSep, numMonteCarlo, numRxFFT]
    magSpec = np.abs(signalFFTshift)**2 # [numbinSep, numMonteCarlo, numRxFFT]

    magSpec1 = np.transpose(magSpec,(2,0,1)) # [numRxFFT, numbinSep, numMonteCarlo]
    magSpec2 = magSpec1.reshape(numAngleFFT, numMonteCarloRuns*numAngSepPoints) # [numRxFFT, numbinSep x numMonteCarlo]
    magSpec3 = np.transpose(magSpec2,(1,0)) # [numbinSep x numMonteCarlo, numRxFFT]

    twoLargestLocalPeaksMatrix = computeTwoLargestLocalMaxima(magSpec3) # [numbinSep x numMonteCarlo, 2]
    estAngDeg = angAxis_deg[twoLargestLocalPeaksMatrix] # [numbinSep x numMonteCarlo, 2]
    estAngSepDeg = np.abs(np.diff(estAngDeg,axis=1)) # [numbinSep x numMonteCarlo]
    estAngSepDeg = estAngSepDeg.reshape(numAngSepPoints, numMonteCarloRuns) # [numbinSep, numMonteCarlo]
    estAngSepArrFFT[count,:,:] = estAngSepDeg

    """ Windowed FFT"""
    windowSignal = signal*np.hanning(numRx)[None,None,:]
    signalFFTWin = np.fft.fft(windowSignal,axis=2,n=numAngleFFT)/numRx # [numbinSep, numMonteCarlo, numRxFFT]
    signalFFTshiftWin = np.fft.fftshift(signalFFTWin,axes=(2,)) # [numbinSep, numMonteCarlo, numRxFFT]
    magSpecWin = np.abs(signalFFTshiftWin)**2 # [numbinSep, numMonteCarlo, numRxFFT]

    magSpec1Win = np.transpose(magSpecWin,(2,0,1)) # [numRxFFT, numbinSep, numMonteCarlo]
    magSpec2Win = magSpec1Win.reshape(numAngleFFT, numMonteCarloRuns*numAngSepPoints) # [numRxFFT, numbinSep x numMonteCarlo]
    magSpec3Win = np.transpose(magSpec2Win,(1,0)) # [numbinSep x numMonteCarlo, numRxFFT]

    twoLargestLocalPeaksMatrixWin = computeTwoLargestLocalMaxima(magSpec3Win) # [numbinSep x numMonteCarlo, 2]
    estAngDegWin = angAxis_deg[twoLargestLocalPeaksMatrixWin] # [numbinSep x numMonteCarlo, 2]
    estAngSepDegWin = np.abs(np.diff(estAngDegWin,axis=1)) # [numbinSep x numMonteCarlo]
    estAngSepDegWin = estAngSepDegWin.reshape(numAngSepPoints, numMonteCarloRuns) # [numbinSep, numMonteCarlo]
    estAngSepArrFFTWin[count,:,:] = estAngSepDegWin


    """ Unwindowed IAA"""
    iaaFormatSignal1 = np.transpose(signal,(2,0,1)) # [numRx, numbinSep, numMonteCarlo]
    iaaFormatSignal2 = iaaFormatSignal1.reshape(numRx, numMonteCarloRuns*numAngSepPoints) # [numRx, numbinSep x numMonteCarlo]
    magSpecIAA = np.zeros((numMonteCarloRuns*numAngSepPoints, numAngleFFT)) # [numbinSep x numMonteCarlo, numRxFFT]
    for ele in np.arange(numMonteCarloRuns*numAngSepPoints):
        spectrum_iaa = iaa_recursive(iaaFormatSignal2[:,ele][:,None], digital_freq_grid, numIAAiterations) # recursive IAA
        magSpec_iaa = np.abs(spectrum_iaa)
        magSpecIAA[ele,:] = magSpec_iaa

    twoLargestLocalPeaksMatrixIAA = computeTwoLargestLocalMaxima(magSpecIAA) # [numbinSep x numMonteCarlo, 2]
    estAngDegIAA = angAxis_deg[twoLargestLocalPeaksMatrixIAA] # [numbinSep x numMonteCarlo, 2]
    estAngSepDegIAA = np.abs(np.diff(estAngDegIAA,axis=1)) # [numbinSep x numMonteCarlo]
    estAngSepDegIAA = estAngSepDegIAA.reshape(numAngSepPoints, numMonteCarloRuns) # [numbinSep, numMonteCarlo]
    estAngSepArrIAA[count,:,:] = estAngSepDegIAA


    count += 1


percent90estAngSepArrFFTWin = np.percentile(estAngSepArrFFTWin,90,axis=2)
percent90estAngSepArrFFT = np.percentile(estAngSepArrFFT,90,axis=2)
percent90estAngSepArrIAA = np.percentile(estAngSepArrIAA,90,axis=2)

percent50estAngSepArrFFTWin = np.percentile(estAngSepArrFFTWin,50,axis=2)
percent50estAngSepArrFFT = np.percentile(estAngSepArrFFT,50,axis=2)
percent50estAngSepArrIAA = np.percentile(estAngSepArrIAA,50,axis=2)


tstop = time.time()

timeMC = tstop - tstart
print('Total time for Monte-Carlo run = {0:.2f} min'.format(timeMC/60))

plt.figure(1,figsize=(20,10),dpi=200)
plt.suptitle('Target SNR = ' + str(binSNRdBArray[0]) + ' dB')
plt.subplot(1,2,1)
plt.title('90 percentile separation')
plt.plot(angSepDeg,percent90estAngSepArrFFTWin.T, '-o', label='Windowed FFT')
plt.plot(angSepDeg,percent90estAngSepArrFFT.T, '-o', label='Unwindowed FFT')
plt.plot(angSepDeg,percent90estAngSepArrIAA.T, '-o', label='IAA')
plt.plot(angSepDeg, angSepDeg, color='k', label='Expectation')
plt.xlabel('GT angular separation (deg)')
plt.ylabel('estimated angular separation (deg)')
plt.axis([angSepDeg[0], angSepDeg[-1], angSepDeg[0], angSepDeg[-1]])
plt.grid(True)
plt.legend()


plt.subplot(1,2,2)
plt.title('50 percentile separation')
plt.plot(angSepDeg,percent50estAngSepArrFFTWin.T, '-o', label='Windowed FFT')
plt.plot(angSepDeg,percent50estAngSepArrFFT.T, '-o', label='Unwindowed FFT')
plt.plot(angSepDeg,percent50estAngSepArrIAA.T, '-o', label='IAA')
plt.plot(angSepDeg, angSepDeg, color='k', label='Expectation')
plt.xlabel('GT angular separation (deg)')
plt.ylabel('estimated angular separation (deg)')
plt.axis([angSepDeg[0], angSepDeg[-1], angSepDeg[0], angSepDeg[-1]])
plt.grid(True)
plt.legend()
