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

numBitsRangeFFTOutput = np.array([12,16,18,20,24,32],dtype=np.int64)
numFracBitsRangeFFTOutput = numBitsRangeFFTOutput-1 # -1 for the integer/sign bit
numTestCases = len(numBitsRangeFFTOutput)

theoretRangeFloorValQuant = -10*np.log10(2**(2*numFracBitsRangeFFTOutput))

""" System parameters"""
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

rfftfpconvfloatAllBitWidths = np.zeros((numTestCases,rfft.shape[0],rfft.shape[1],rfft.shape[2]),dtype=np.complex128)
for ele in range(numTestCases):
    rfftfp = np.floor(rfft.real * 2**numFracBitsRangeFFTOutput[ele] + 0*0.5) + \
        1j*np.floor(rfft.imag * 2**numFracBitsRangeFFTOutput[ele] + 0*0.5)

    rfftfpconvfloat = rfftfp/(2**numFracBitsRangeFFTOutput[ele])
    rfftfpconvfloatAllBitWidths[ele,:,:,:] = rfftfpconvfloat


flagRfftPowMean = 1
if flagRfftPowMean:
    rfftSpecdB = 10*np.log10(np.mean(np.abs(rfft)**2,axis=(1,2)))

    rfftfpconvfloatSpec = np.mean(np.abs(rfftfpconvfloatAllBitWidths)**2,axis=(2,3))
    rfftfpconvfloatSpecdB = 10*np.log10(rfftfpconvfloatSpec)
else:
    rfftSpecdB = 10*np.log10(np.abs(rfft[:,0,0])**2)

    rfftfpconvfloatSpecdB = 10*np.log10(np.abs(rfftfpconvfloatAllBitWidths)**2)
    rfftfpconvfloatSpecdB = rfftfpconvfloatSpecdB[:,:,0,0]


dfft = np.fft.fft(rfft,axis=1)/numChirps

""" Perform DFFT on the quantized RFFT output"""
dfftfp = np.fft.fft(rfftfpconvfloatAllBitWidths,axis=2)/numChirps

dfftfpconvfloatAllBitWidths = np.zeros((numTestCases,rfft.shape[0],rfft.shape[1],rfft.shape[2]),dtype=np.complex128)
for ele in range(numTestCases):

    dfftfp_ = np.floor(dfftfp[ele,:,:,:].real * 2**numFracBitsRangeFFTOutput[ele] + 0*0.5) + \
        1j*np.floor(dfftfp[ele,:,:,:].imag * 2**numFracBitsRangeFFTOutput[ele] + 0*0.5)

    dfftfpconvfloat = dfftfp_/(2**numFracBitsRangeFFTOutput[ele])
    dfftfpconvfloatAllBitWidths[ele,:,:,:] = dfftfpconvfloat


flagDfftPowMean = 1
if flagDfftPowMean:
    dfftSpecdB = 10*np.log10(np.mean(np.abs(dfft[targetRangeBins,:,:])**2,axis=(2)))

    dfftfpconvfloatSpec = np.mean(np.abs(dfftfpconvfloatAllBitWidths[:,targetRangeBins,:,:])**2,axis=(3))
    dfftfpconvfloatSpecdB = 10*np.log10(dfftfpconvfloatSpec)
else:
    dfftSpecdB = 10*np.log10(np.abs(dfft[targetRangeBins,:,0])**2)

    dfftfpconvfloatSpecdB = 10*np.log10(np.abs(dfftfpconvfloatAllBitWidths)**2)
    dfftfpconvfloatSpecdB = dfftfpconvfloatSpecdB[:,targetRangeBins,:,0]

doppnoiseFloordB = noiseFloordB - 10*np.log10(numChirps)



plt.figure(1,figsize=(20,10))
plt.suptitle('Range spectrum')
for ele in range(numTestCases):
    plt.subplot(2,3,ele+1)
    plt.title('{} bit RFFT'.format(numBitsRangeFFTOutput[ele]))
    plt.plot(rfftSpecdB,label='Without FFT quantization')
    plt.plot(rfftfpconvfloatSpecdB[ele,:],lw=2,alpha=0.5,label='With {} bit FFT quantization'.format(numBitsRangeFFTOutput[ele]))
    plt.axhline(noiseFloordB,color='k',ls='dotted',label='Programmed noise floor')
    plt.axhline(theoretRangeFloorValQuant[ele],color='k',ls='dashed',label='Quant floor due to {} bit RFFT'.format(numBitsRangeFFTOutput[ele]))
    plt.xlabel('Range bins')
    plt.ylabel('dBm')
    plt.legend()
    plt.grid(True)
    plt.ylim(min(min(theoretRangeFloorValQuant),noiseFloordB)-10, 10)


doppLegend = ['Target 1 without Dopp FFT quant', 'Target 2 without Dopp FFT quant',
              'Target 1 with Dopp FFT quant', 'Target 2 with Dopp FFT quant', 'Expected Dopp noise floor']

plt.figure(2,figsize=(20,10))
plt.suptitle('Doppler spectrum')
for ele in range(numTestCases):
    plt.subplot(2,3,ele+1)
    plt.title('{} bit DFFT'.format(numBitsRangeFFTOutput[ele]))
    plt.plot(dfftSpecdB.T)
    plt.plot(dfftfpconvfloatSpecdB[ele,:,:].T,lw=2,alpha=0.5)
    plt.axhline(doppnoiseFloordB,color='k',ls='dotted')
    plt.axhline(theoretRangeFloorValQuant[ele],color='k',ls='dashed')
    plt.xlabel('Doppler bins')
    plt.ylabel('dBm')
    doppLegendFull = doppLegend + ['Quant floor due to {} bit DFFT'.format(numBitsRangeFFTOutput[ele])]
    plt.legend(doppLegendFull)
    plt.grid(True)
    plt.ylim(min(min(theoretRangeFloorValQuant),doppnoiseFloordB)-10, 10)

