# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 23:04:53 2021

@author: saiguna
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
# np.random.seed(0)

numSamples = 1024
numFFTBins = 1024 #8192
print('Num FFT points = {}'.format(numFFTBins))
fftOverSamplingFact = numFFTBins//numSamples
thermalNoise = -174 # dBm/Hz
noiseFigure = 10 # dB
baseBandgain = 34 #dB
adcSamplingRate = 35e6 # 35 MHz
dBFs_to_dBm = 10
totalNoisePower_dBm = thermalNoise + noiseFigure + baseBandgain + 10*np.log10(adcSamplingRate)
totalNoisePower_dBFs = totalNoisePower_dBm - 10
noiseFloor_perBin = totalNoisePower_dBFs - 10*np.log10(numFFTBins) # dBFs/bin
# noiseFloor_perBin = -100 # dBFs
noisePower_perBin = 10**(noiseFloor_perBin/10)
binSNR = 20 # dB

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

estSignalPower = 10*np.log10(np.sum(spectrum[fftOverSamplingFact*(objectRangeBinInt-3):fftOverSamplingFact*(objectRangeBinInt+4)]))
estNoiseFloor = 10*np.log10(np.mean(np.sort(spectrum)[0:numFFTBins//2]))


print('Estimated Signal Power = {} dBFs'.format(np.round(estSignalPower)))
print('True Noise power/bin = {} dBFs'.format(np.round(noiseFloor_perBin)))
print('Estimated Noise power/bin = {} dBFs'.format(np.round(estNoiseFloor)))
print('Estimated SNR = {}'.format(np.round(estSignalPower-estNoiseFloor)))

signalSpectrumdBm = 20*np.log10(np.abs(fftsignal)) + dBFs_to_dBm

plt.figure(1,figsize=(20,10))
plt.plot(signalSpectrumdBm)
plt.ylabel('dBm')
plt.grid(True)






