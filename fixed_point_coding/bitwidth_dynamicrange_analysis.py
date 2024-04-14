# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 23:06:16 2024

@author: Sai Gunaranjan
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

numADCBits = 16
numADCFracBits = numADCBits-1

numBitsRangeFFTOutput = 12
numFracBitsRangeFFTOutput = numBitsRangeFFTOutput-1 # -1 for the integer/sign bit

numADCSamples = 2048
numRangeSamples = numADCSamples//2
numChirps = 512
numRxs = 4
""" Noise parameters"""
thermalFloor = -174 # dBm/Hz
rxGain = 38 #dB
NF = 9 # dB
adcSamplRate = 56.25e6 # 56.25 MHz
rbw = 10*np.log10(adcSamplRate/numADCSamples)
# noiseFloordB = thermalFloor + rbw + rxGain + NF
noiseFloordB = -90 # bin snr
noise_power_db = noiseFloordB + 10*np.log10(numADCSamples)
noise_variance = 10**(noise_power_db/10)
noise_sigma = np.sqrt(noise_variance)

object_snr = np.array([60,10]) #np.array([60,-10])
weights = np.sqrt(10**((noiseFloordB + object_snr)/10))

targetRangeBins = np.array([20,800])
targetDopplerBins = np.array([0,200])
targetAngleBins = np.array([0,2])

rangeSignal = np.exp(1j*2*np.pi*targetRangeBins[:,None]*np.arange(numADCSamples)[None,:]/numADCSamples)
dopplerSignal = np.exp(1j*2*np.pi*targetDopplerBins[:,None]*np.arange(numChirps)[None,:]/numChirps)
angleSignal = np.exp(1j*2*np.pi*targetAngleBins[:,None]*np.arange(numRxs)[None,:]/numRxs)

signalComponents = weights[:,None,None,None] * rangeSignal[:,:,None,None] * dopplerSignal[:,None,:,None] * angleSignal[:,None,None,:]
cleanSignal = np.sum(signalComponents,axis=0)

noise = (noise_sigma/np.sqrt(2))*np.random.randn(numADCSamples * numChirps * numRxs) + \
    1j*(noise_sigma/np.sqrt(2))*np.random.randn(numADCSamples * numChirps * numRxs)

noise = noise.reshape(numADCSamples,numChirps,numRxs)

radarSignal = cleanSignal + noise

adcRadarSignal = np.floor(radarSignal.real * 2**numADCFracBits + 0.5) + \
    1j*np.floor(radarSignal.imag * 2**numADCFracBits + 0.5)

rfft = np.fft.fft(adcRadarSignal,axis=0)/(numADCSamples* 2**numADCFracBits)
rfft = rfft[0:numRangeSamples,:,:]



rfftfp = np.floor(rfft.real * 2**numFracBitsRangeFFTOutput + 0*0.5) + \
    1j*np.floor(rfft.imag * 2**numFracBitsRangeFFTOutput + 0*0.5)

rfftfpconvfloat = rfftfp/(2**numFracBitsRangeFFTOutput)
rfftfpconvfloatSpec = np.mean(np.abs(rfftfpconvfloat)**2,axis=(1,2))

flagRfftPowMean = 1

if flagRfftPowMean:
    rfftSpecdB = 10*np.log10(np.mean(np.abs(rfft)**2,axis=(1,2)))

    rfftfpconvfloatSpecdB = 10*np.log10(rfftfpconvfloatSpec)
else:
    rfftSpecdB = 10*np.log10(np.abs(rfft[:,0,0])**2)

    rfftfpconvfloatSpecdB = 10*np.log10(np.abs(rfftfpconvfloat)**2)
    rfftfpconvfloatSpecdB = rfftfpconvfloatSpecdB[:,0,0]






theoretRangeFloorValQuant = 10*np.log10(2**(2*-numFracBitsRangeFFTOutput))

plt.figure(1,figsize=(20,10),dpi=200)
plt.title('Range spectrum')
plt.plot(rfftSpecdB,label='Without FFT quantization')
plt.plot(rfftfpconvfloatSpecdB,lw=2,alpha=0.5,label='With {} bit FFT output'.format(numBitsRangeFFTOutput))
plt.axhline(theoretRangeFloorValQuant,color='k',ls='dashed',label='Quant floor due to {} bit RFFT quantization'.format(numBitsRangeFFTOutput))
plt.axhline(noiseFloordB,color='k',ls='dotted',label='Programmed noise floor')
plt.xlabel('Range bins')
plt.ylabel('dBm')
plt.legend()
plt.grid(True)
plt.ylim(min(theoretRangeFloorValQuant,noiseFloordB)-10, 10)