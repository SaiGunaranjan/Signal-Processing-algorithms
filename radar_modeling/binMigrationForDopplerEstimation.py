# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 23:04:53 2021

@author: saiguna
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

noiseFloor_perBin = -100 # dBFs
noisePower_perBin = 10**(noiseFloor_perBin/10)
binSNR = 20 # dB
numSamples = 1024
numFFTBins = 1024
totalNoisePower = noisePower_perBin*numFFTBins # sigmasquare totalNoisePower
sigma = np.sqrt(totalNoisePower)
signalPowerdBFs = noiseFloor_perBin + binSNR
signalPower = 10**(signalPowerdBFs/10)
signalAmplitude = np.sqrt(signalPower)
signalPhase = np.exp(1j*np.random.uniform(-np.pi, np.pi))
signalphasor = signalAmplitude*signalPhase
objectRange_m = 10
chirpBW = 4e9
lightSpeed = 3e8
rangeRes = lightSpeed/(2*chirpBW)
objectRangeBinInt = np.int(objectRange_m/rangeRes)
objectRangeBin = objectRangeBinInt + np.random.uniform(-0.5,0.5)
signal = signalphasor*np.exp(1j*((2*np.pi*objectRangeBin)/numSamples)*np.arange(numSamples))
noise = (sigma/np.sqrt(2))*np.random.randn(numSamples) + 1j*(sigma/np.sqrt(2))*np.random.randn(numSamples)

signal_noise = signal + noise

receivedSignal = signal_noise*np.hanning(numSamples)

fftsignal = np.fft.fft(receivedSignal, n=numFFTBins)/numSamples
fftsignal = fftsignal[0:numFFTBins//2]
spectrum = np.abs(fftsignal)**2

estSignalPower = 10*np.log10(np.sum(spectrum[objectRangeBinInt-3:objectRangeBinInt+4]))
estNoiseFloor = 10*np.log10(np.mean(np.sort(spectrum)[0:400]))

print('Estimted SNR = {}'.format(np.round(estSignalPower-estNoiseFloor)))

plt.plot(20*np.log10(np.abs(fftsignal)))
plt.grid(True)






