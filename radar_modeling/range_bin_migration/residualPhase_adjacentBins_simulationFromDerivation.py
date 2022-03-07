# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 18:19:40 2022

@author: Sai Gunaranjan Pelluri
"""

""" This script illustrates that the phase jump introduced due to range bin migration is equal to pi-pi/N radians,
where N is the number of signal samples in a given chirp. The theoretical derivation for this results is available in the below link:
    https://saigunaranjan.atlassian.net/wiki/spaces/RM/pages/1245185/Phase+Jump+due+to+range+bin+migration
    This script simulates the same way that has been described in the above link derivation.
    """


import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

""" System Parameters """

""" Below are the system parameters used for the 256v2J platform"""
numSamples = 2048
numFFTBins = 2048 #8192
thermalNoise = -174 # dBm/Hz
noiseFigure = 10 # dB
baseBandgain = 34 #dB
adcSamplingRate = 56.25e6 # 56.25 MHz
dBFs_to_dBm = 10
binSNR = 40 # dB


""" Chirp Parameters"""
numChirps = 168


fftOverSamplingFact = numFFTBins//numSamples
adcSamplingTime = 1/adcSamplingRate # seconds
chirpOnTime = numSamples*adcSamplingTime #39.2e-6

totalNoisePower_dBm = thermalNoise + noiseFigure + baseBandgain + 10*np.log10(adcSamplingRate)
totalNoisePower_dBFs = totalNoisePower_dBm - 10
noiseFloor_perBin = totalNoisePower_dBFs - 10*np.log10(numFFTBins) # dBFs/bin
noisePower_perBin = 10**(noiseFloor_perBin/10)


totalNoisePower = noisePower_perBin*numFFTBins # sigmasquare totalNoisePower
sigma = np.sqrt(totalNoisePower)
signalPowerdBFs = noiseFloor_perBin + binSNR
signalPower = 10**(signalPowerdBFs/10)
signalAmplitude = np.sqrt(signalPower)
signalPhase = np.exp(1j*np.random.uniform(-np.pi, np.pi))
signalphasor = signalAmplitude*signalPhase

noise = (sigma/np.sqrt(2))*np.random.randn(numSamples*numChirps) + 1j*(sigma/np.sqrt(2))*np.random.randn(numSamples*numChirps)
noise = noise.reshape(numSamples, numChirps)

freqBin = 0
binoffset = np.linspace(0 ,1, numChirps) # binoffset is same as alpha in the above link
freqBinInt = int(freqBin)
Signal = signalphasor*np.exp(1j*((2*np.pi*(freqBin+binoffset[None,:]))/numSamples)*np.arange(numSamples)[:,None])

Signal_bin = Signal.copy()
Signal_binPlus1 = signalphasor*np.exp(1j*((2*np.pi*(freqBin+(binoffset[None,:]-1)))/numSamples)*np.arange(numSamples)[:,None])


Signal_bin = Signal_bin + noise
Signal_binPlus1 = Signal_binPlus1 + noise

Signal_bin = Signal_bin*np.hanning(numSamples)[:,None]
Signal_binPlus1 = Signal_binPlus1*np.hanning(numSamples)[:,None]

Phasor_bin = np.sum(Signal_bin,axis=0)
Phasor_binPlus1 = np.sum(Signal_binPlus1,axis=0)

plt.figure(1, figsize=(20,10))
plt.suptitle('Target frequency is made to move from bin to bin+1 from 1st chirp to last chirp')
plt.subplot(1,3,1)
plt.title('Residual Magnitude')
plt.plot(binoffset, 20*np.log10(np.abs(Phasor_bin)),label='Residual Magnitude at Bin')
plt.plot(binoffset, 20*np.log10(np.abs(Phasor_binPlus1)),label='Residual Magnitude at Bin + 1')
plt.legend()
plt.xlabel('bin offset')
plt.ylabel('Magnitude (dB)')
plt.grid(True)

phase_bin = np.unwrap(np.angle(Phasor_bin))
phase_binPlus1 = np.unwrap(np.angle(Phasor_binPlus1))

plt.subplot(1,3,2)
plt.title('Residual Phase')
plt.plot(binoffset, phase_bin,label='Residual Phase at Bin')
plt.plot(binoffset, phase_binPlus1,label='Residual Phase at Bin + 1')
plt.legend()
plt.xlabel('bin offset')
plt.ylabel('Phase (rad)')
plt.grid(True)


plt.subplot(1,3,3)
plt.title('Residual Phase @ bin - Residual Phase @ bin+1')
plt.plot(binoffset, np.abs(phase_bin - phase_binPlus1))
plt.axhline(np.pi-np.pi/numSamples, label='pi - pi/N',color='k')
plt.ylim([0,5])
plt.xlabel('bin offset')
plt.ylabel('Phase (rad)')
plt.legend()
plt.grid(True)