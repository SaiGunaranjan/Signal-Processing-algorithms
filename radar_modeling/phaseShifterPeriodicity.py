# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:36:29 2021

@author: saiguna
"""


import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

numBitsPhaseShifter = 5
numPhaseCodes = 2**numBitsPhaseShifter
DNL = 360/(numPhaseCodes) # DNL in degrees
phaseStepPerRamp_deg = 29 # With 30 deg we see periodicity since 30 divides 360 but with say 29 deg, it doesnt divide 360 and hence periodicity is significantly reduced
numRamps = 280
phaseShifterCodes = DNL*np.arange(numPhaseCodes)
phaseShifterNoise = np.random.uniform(-DNL/2, DNL/2, numPhaseCodes)
phaseShifterCodes_withNoise = phaseShifterCodes + phaseShifterNoise

rampPhaseIdeal_deg = phaseStepPerRamp_deg*np.arange(numRamps)
rampPhaseIdeal_degWrapped = np.mod(rampPhaseIdeal_deg, 360)

phaseCodesIndexToBeApplied = np.argmin(np.abs(rampPhaseIdeal_degWrapped[:,None] - phaseShifterCodes_withNoise[None,:]),axis=1)
phaseCodesToBeApplied = phaseShifterCodes_withNoise[phaseCodesIndexToBeApplied]

phaseCodesToBeApplied_rad = phaseCodesToBeApplied/180 * np.pi

signal = np.exp(1j*phaseCodesToBeApplied_rad)

signalWindowed = signal*np.hanning(numRamps)
signalFFT = np.fft.fft(signalWindowed)/numRamps
signalFFTShift = np.fft.fftshift(signalFFT)

noiseFloorSetByDNL = 10*np.log10((DNL/180 *np.pi)**2/12) - 10*np.log10(numRamps)

plt.figure(1, figsize=(20,10))
plt.title('Doppler Spectrum: Floor set by DNL = ' + str(np.round(noiseFloorSetByDNL)) + ' dB/bin')
plt.plot(20*np.log10(np.abs(signalFFTShift)))
plt.xlabel('Bins')
plt.ylabel('Power dBFs')
plt.grid(True)



