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
# print('Num FFT points = {}'.format(numFFTBins))
fftOverSamplingFact = numFFTBins//numSamples
thermalNoise = -174 # dBm/Hz
noiseFigure = 10 # dB
baseBandgain = 34 #dB
adcSamplingRate = 56.25e6 # 56.25 MHz
adcSamplingTime = 1/adcSamplingRate # seconds
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
chirpBW = 4e9 # Hz
lightSpeed = 3e8 # m/s
rangeRes = lightSpeed/(2*chirpBW)
rangeAxis_m = np.arange(numFFTBins//2)*rangeRes
objectRangeBin = objectRange_m/rangeRes
objectRangeBinInt = np.int(objectRangeBin)
rangeSignal = signalphasor*np.exp(1j*((2*np.pi*objectRangeBin)/numSamples)*np.arange(numSamples))

""" Chirp Parameters"""
interRampTime = 44e-6 # us
chirpSamplingRate = 1/interRampTime
chirpOnTime = 39.2e-6
chirpSlope = chirpBW/chirpOnTime
chirpStartFreq = 77e9 # Giga Hz
chirpCentreFreq = chirpStartFreq + chirpBW/2
lamda = lightSpeed/chirpCentreFreq
maxVelBaseband_mps = (chirpSamplingRate/2) * (lamda/2) # m/s
numChirps = 168
objectVelocity_mps = 40 # m/s
objectVelocity_baseBand_mps = np.mod(objectVelocity_mps, maxVelBaseband_mps)
if (objectVelocity_baseBand_mps >= maxVelBaseband_mps/2):
    objectVelocity_baseBand_mps = objectVelocity_baseBand_mps - maxVelBaseband_mps

velocityRes = (chirpSamplingRate/numChirps) * (lamda/2)
objectVelocityBin = objectVelocity_baseBand_mps/velocityRes
objectVelocityInt = np.int(objectVelocityBin)

dopplerSignal = np.exp(1j*((2*np.pi*objectVelocityBin)/numChirps)*np.arange(numChirps))

rangeBinMigrationTerm = \
    np.exp(1j*2*np.pi*chirpSlope*(2*objectVelocity_mps/lightSpeed)*interRampTime*adcSamplingTime*np.arange(numSamples)[:,None]*np.arange(numChirps)[None,:])

radarSignal = rangeSignal[:,None] * dopplerSignal[None,:] * rangeBinMigrationTerm

noise = (sigma/np.sqrt(2))*np.random.randn(numSamples*numChirps) + 1j*(sigma/np.sqrt(2))*np.random.randn(numSamples*numChirps)
noise = noise.reshape(numSamples, numChirps)
signal_noise = radarSignal + noise

receivedSignal = signal_noise*np.hanning(numSamples)[:,None]

rangeFFTSignal = np.fft.fft(receivedSignal, n=numFFTBins, axis=0)/numSamples
rangeFFTSignal = rangeFFTSignal[0:numFFTBins//2,:]


# spectrum = np.abs(rangeFFTSignal)**2
# estSignalPower = 10*np.log10(np.sum(spectrum[fftOverSamplingFact*(objectRangeBinInt-3):fftOverSamplingFact*(objectRangeBinInt+4)]))
# estNoiseFloor = 10*np.log10(np.mean(np.sort(spectrum)[0:numFFTBins//2]))


# print('Estimated Signal Power = {} dBFs'.format(np.round(estSignalPower)))
# print('True Noise power/bin = {} dBFs'.format(np.round(noiseFloor_perBin)))
# print('Estimated Noise power/bin = {} dBFs'.format(np.round(estNoiseFloor)))
# print('Estimated SNR = {}'.format(np.round(estSignalPower-estNoiseFloor)))

signalSpectrumdBm = 20*np.log10(np.abs(rangeFFTSignal)) + dBFs_to_dBm

plt.figure(1,figsize=(20,10))
plt.plot(rangeAxis_m, signalSpectrumdBm)
plt.axvline(objectRange_m, color = 'k')
plt.xlabel('Range (m)')
plt.ylabel('dBm')
plt.grid(True)






