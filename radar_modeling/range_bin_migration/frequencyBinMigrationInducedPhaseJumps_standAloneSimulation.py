# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 13:33:51 2022

@author: Sai Gunaranjan Pelluri
"""

""" Stand alone script to model frequency bin migration/movement and observe the phase jump introduced due to migration"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
# np.random.seed(0)

""" System Parameters """

""" Below are the system parameters used for the 256v2J platform"""
numSamples = 2048
numFFTBins = 2048 #8192
thermalNoise = -174 # dBm/Hz
noiseFigure = 10 # dB
baseBandgain = 34 #dB
adcSamplingRate = 56.25e6 # 56.25 MHz
dBFs_to_dBm = 10
binSNR = 20 # dB


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



freqBin = 230
binoffset = np.linspace(0 ,1, numChirps)
freqBinInt = int(freqBin)
Signal = signalphasor*np.exp(1j*((2*np.pi*(freqBin+binoffset[None,:]))/numSamples)*np.arange(numSamples)[:,None])

noise = (sigma/np.sqrt(2))*np.random.randn(numSamples*numChirps) + 1j*(sigma/np.sqrt(2))*np.random.randn(numSamples*numChirps)
noise = noise.reshape(numSamples, numChirps)

signal_noise = Signal + 0*noise
receivedSignal = signal_noise*np.hanning(numSamples)[:,None]

FFTSignal = np.fft.fft(receivedSignal, n=numFFTBins, axis=0)/numSamples
FFTSignal = FFTSignal[0:numFFTBins//2,:]
signalSpectrumdBm = 20*np.log10(np.abs(FFTSignal)) + dBFs_to_dBm

phase_refbinMinus1 = np.unwrap(np.angle(FFTSignal[freqBin-1]))
phase_refBin = np.unwrap(np.angle(FFTSignal[freqBin]))
phase_refbinPlus1 = np.unwrap(np.angle(FFTSignal[freqBin+1]))


plt.figure(1,figsize=(20,10))
plt.suptitle('Target frequency is made to move from bin to bin+1 from 1st chirp to last chirp')
plt.subplot(1,3,1)
plt.title('Magnitude Spectrum')
plt.plot(signalSpectrumdBm)
plt.axvline(freqBin, color = 'k')
plt.xlabel('FFT bin ')
plt.ylabel('dBm')
plt.xlim([freqBin-10, freqBin+11])
plt.grid(True)


plt.subplot(1,3,2)
plt.title('Residual bin migration phase across chirps')
# plt.plot(phase_refbinMinus1, label='bin - 1')
plt.plot(phase_refBin, label='bin')
plt.plot(phase_refbinPlus1, label='bin + 1')
plt.legend()

plt.ylabel('Phase (rad)')
plt.grid(True)



plt.subplot(1,3,3)
plt.title('Phase(bin) - phase(bin+1) for all chirps')
plt.plot(np.abs(phase_refBin-phase_refbinPlus1), linewidth=4, alpha=0.7, label='Phase(bin) - phase(bin+1)')
plt.axhline(np.pi, label='3.14',color='k')
plt.grid(True)
plt.legend()
plt.xlabel('Chirp number');
plt.ylabel('Phase (rad)')
plt.ylim([3.11,3.16])


targetRangeBins = np.argmax(signalSpectrumdBm,axis=0)
DopplerPhaseMigratedRangeBins = FFTSignal[targetRangeBins[None,:],np.arange(numChirps)][0,:]

binDelta = np.abs(targetRangeBins[1::] - targetRangeBins[0:-1])
tempVar = binDelta*np.pi # binDelta*(np.pi-np.pi/numSamples)
binMigrationPhaseCorrTerm = np.zeros((targetRangeBins.shape),dtype=np.float32)
binMigrationPhaseCorrTerm[1::] = tempVar
binMigrationPhasorCorrTerm = np.exp(-1j*np.cumsum(binMigrationPhaseCorrTerm))

DopplerPhaseMigratedRangeBins_PhaseCorrected = DopplerPhaseMigratedRangeBins*binMigrationPhasorCorrTerm

plt.figure(2,figsize=(20,10))
# plt.suptitle('Object moving at ' + str(objectVelocity_mps) + ' m/s;' + 'chirp BW =' + str(int(chirpBW/1e9)) + 'GHz')
plt.subplot(1,3,1)
plt.title('Range peak across chirps')
plt.plot(targetRangeBins, '-o')
plt.xlabel('Chirp Number')
plt.ylabel('Range Bin')
plt.grid(True)

plt.subplot(1,3,2)
plt.title('Residual bin migration phase across chirps at peak range bin')
plt.plot(np.unwrap(np.angle(DopplerPhaseMigratedRangeBins)), linewidth=2)
plt.xlabel('Chirp Number')
plt.ylabel('Phase (rad)')
plt.grid(True)

plt.subplot(1,3,3)
plt.title('Residual bin migration phase across chirps at peak range bin after pi correction')
plt.plot(np.unwrap(np.angle(DopplerPhaseMigratedRangeBins_PhaseCorrected)), linewidth=2)
plt.xlabel('Chirp Number')
plt.ylabel('Phase (rad)')
plt.grid(True)
