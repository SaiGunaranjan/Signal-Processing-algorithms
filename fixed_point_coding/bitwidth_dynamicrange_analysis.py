# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 23:06:16 2024

@author: Sai Gunaranjan
"""

"""
1. Plot the theoretical noise floor/underflow for different bitwidths post range fft, doppler fft, angle fft
2. Plot measured snr vs actual snr for differnt bitwidths and a detection SNR. This should be for a given noise floor,
system gain.

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
dbFstodBm = 10
noiseFloordBm = thermalFloor + rbw + rxGain + NF # dBm
noiseFloordB = noiseFloordBm - dbFstodBm # dBFs
# noiseFloordB = -90 # bin snr
noise_power_db = noiseFloordB + 10*np.log10(numADCSamples)
noise_variance = 10**(noise_power_db/10)
noise_sigma = np.sqrt(noise_variance)
""" """

systemGain = 10*np.log10(numChirps) + 10*np.log10(numRxs)
detectionSNR = 18 # dB
snrPostRFFT = 18 - systemGain
minSignalPowerRFFT = snrPostRFFT + noiseFloordB

print('\nMin signal power at RFFT should be = {0:.1f} dB to be detected post {1} point DFFT, \
{2} point channel comb. for a detection SNR of {3} dB\n'.format(minSignalPowerRFFT,numChirps,numRxs,detectionSNR))

minTargetRangeBin = 20
numTargets = 8#np.ceil(np.log2(numRangeSamples/minTargetRangeBin)).astype(np.int32) # Targets at double the previous distance
object_snr = np.linspace(60,-15,numTargets)#np.array([60,10]) #np.array([60,-10])
weights = np.sqrt(10**((noiseFloordB + object_snr)/10))

targetRangeBins = np.int32(np.linspace(minTargetRangeBin,numRangeSamples-1,numTargets))#minTargetRangeBin* 2**(np.arange(numTargets))#np.array([20,800])
targetDopplerBins = np.random.randint(0,numChirps,size=numTargets)#np.array([0,200])
targetAngleBins = np.random.randint(0,numRxs,size=numTargets)#np.array([0,2])

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
    secondSmallestVal = np.unique(dfftfpconvfloatSpec)[1]
    dfftfpconvfloatSpec[dfftfpconvfloatSpec==0] = secondSmallestVal
    dfftfpconvfloatSpecdB = 10*np.log10(dfftfpconvfloatSpec)
else:
    dfftSpecdB = 10*np.log10(np.abs(dfft[targetRangeBins,:,0])**2)

    dfftfpconvfloatAllBitWidthsMagSq = np.abs(dfftfpconvfloatAllBitWidths)**2
    secondSmallestVal = np.unique(dfftfpconvfloatAllBitWidthsMagSq)[1]
    dfftfpconvfloatAllBitWidthsMagSq[dfftfpconvfloatAllBitWidthsMagSq==0] = secondSmallestVal
    dfftfpconvfloatSpecdB = 10*np.log10(dfftfpconvfloatAllBitWidthsMagSq)
    dfftfpconvfloatSpecdB = dfftfpconvfloatSpecdB[:,targetRangeBins,:,0]

doppnoiseFloordB = noiseFloordB - 10*np.log10(numChirps) # because we are normalizing, the signal power stays same and noise power drops by 10logN


rxfft = np.fft.fft(dfft,axis=2)/numRxs

""" Perform RxFFT on the quantized DFFT output"""
rxfftfp = np.fft.fft(dfftfpconvfloatAllBitWidths,axis=3)/numRxs

rxfftfpconvfloatAllBitWidths = np.zeros((numTestCases,rfft.shape[0],rfft.shape[1],rfft.shape[2]),dtype=np.complex128)
for ele in range(numTestCases):

    rxfftfp_ = np.floor(rxfftfp[ele,:,:,:].real * 2**numFracBitsRangeFFTOutput[ele] + 0*0.5) + \
        1j*np.floor(rxfftfp[ele,:,:,:].imag * 2**numFracBitsRangeFFTOutput[ele] + 0*0.5)

    rxfftfpconvfloat = rxfftfp_/(2**numFracBitsRangeFFTOutput[ele])
    rxfftfpconvfloatAllBitWidths[ele,:,:,:] = rxfftfpconvfloat


rxfftSpecdB = 10*np.log10(np.abs(rxfft[targetRangeBins,targetDopplerBins,:])**2)

rxfftfpconvfloatAllBitWidthsMagSq = np.abs(rxfftfpconvfloatAllBitWidths)**2
secondSmallestVal = np.unique(rxfftfpconvfloatAllBitWidthsMagSq)[1]
rxfftfpconvfloatAllBitWidthsMagSq[rxfftfpconvfloatAllBitWidthsMagSq==0] = secondSmallestVal
rxfftfpconvfloatSpecdB = 10*np.log10(rxfftfpconvfloatAllBitWidthsMagSq)
rxfftfpconvfloatSpecdB = rxfftfpconvfloatSpecdB[:,targetRangeBins,targetDopplerBins,:]

rxnoiseFloordB = doppnoiseFloordB - 10*np.log10(numRxs) # because we are normalizing, the signal power stays same and noise power drops by 10logN


trueSignalPower = rxfftSpecdB[np.arange(numTargets),targetAngleBins]
trueNoisePower = rxnoiseFloordB
trueSNR = object_snr+systemGain #trueSignalPower - trueNoisePower
trueSNR = np.round(trueSNR,2)

measSignalPower = rxfftfpconvfloatSpecdB[:,np.arange(numTargets),targetAngleBins]
measNoisePower = np.maximum(theoretRangeFloorValQuant,rxnoiseFloordB)
measSNR = measSignalPower - measNoisePower[:,None]
measSNR[measSNR<=0] = -1000 # signal lost in the floor quantization. These targets are lost and will not be detected.




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
    plt.ylabel('dBFs')
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
    plt.plot(dfftSpecdB[np.array([0,-1]),:].T)
    plt.plot(dfftfpconvfloatSpecdB[ele,np.array([0,-1]),:].T,lw=2,alpha=0.5)
    plt.axhline(doppnoiseFloordB,color='k',ls='dotted')
    plt.axhline(theoretRangeFloorValQuant[ele],color='k',ls='dashed')
    plt.xlabel('Doppler bins')
    plt.ylabel('dBFs')
    doppLegendFull = doppLegend + ['Quant floor due to {} bit DFFT'.format(numBitsRangeFFTOutput[ele])]
    plt.legend(doppLegendFull)
    plt.grid(True)
    plt.ylim(min(min(theoretRangeFloorValQuant),doppnoiseFloordB)-10, 10)

if 0:
    angLegend = ['Target 1 without Rx FFT quant', 'Target 2 without Rx FFT quant',
                  'Target 1 with Rx FFT quant', 'Target 2 with Rx FFT quant', 'Expected noise floor']
    plt.figure(3,figsize=(20,10))
    plt.suptitle('Angle spectrum')
    for ele in range(numTestCases):
        plt.subplot(2,3,ele+1)
        plt.title('{} bit DFFT'.format(numBitsRangeFFTOutput[ele]))
        plt.plot(rxfftSpecdB[np.array([0,-1]),:].T)
        plt.plot(rxfftfpconvfloatSpecdB[ele,np.array([0,-1]),:].T,lw=2,alpha=0.5)
        plt.axhline(rxnoiseFloordB,color='k',ls='dotted')
        plt.axhline(theoretRangeFloorValQuant[ele],color='k',ls='dashed')
        plt.xlabel('Angle bins')
        plt.ylabel('dBFs')
        angLegendFull = angLegend + ['Quant floor due to {} bit DFFT'.format(numBitsRangeFFTOutput[ele])]
        plt.legend(angLegendFull)
        plt.grid(True)
        plt.ylim(min(min(theoretRangeFloorValQuant),rxnoiseFloordB)-10, 10)


plt.figure(4,figsize=(20,10),dpi=200)
plt.title('Quantization Noise floor(Minimum representable power) vs bitwidth')
plt.plot(numBitsRangeFFTOutput,theoretRangeFloorValQuant,'-o')
plt.axhline(noiseFloordB,label='True RFFT floor',color='k',ls='dashed')
plt.axhline(doppnoiseFloordB,label='True DFFT floor',color='k',ls='dotted')
plt.axhline(rxnoiseFloordB,label='True Rx FFT floor',color='k',ls='dashdot')
plt.xlabel('bitwidth')
plt.ylabel('dBFs')
plt.legend()
plt.grid(True)
plt.xticks(numBitsRangeFFTOutput)


bitwidthLeg = [str(ele) + ' bit quant.' for ele in numBitsRangeFFTOutput]
bitwidthLeg.append('GT')
bitwidthLeg.append('Detection SNR = {} dB'.format(detectionSNR))
plt.figure(5,figsize=(20,10),dpi=200)
plt.title('Measured SNR vs True SNR (post detection gain of {} chirps, {} Rxs)'.format(numChirps,numRxs))
# plt.plot(trueSNR,measSNR.T,'-o')
plt.plot(trueSNR,measSNR[0:-2,:].T,'-o')
plt.plot(trueSNR,measSNR[-2,:],'-o',lw=6,alpha=0.5)
plt.plot(trueSNR,measSNR[-1,:],'-o')
plt.plot(trueSNR,trueSNR,'^',color='k')
plt.axhline(detectionSNR,ls='dashed',color='k')
plt.xlabel('True SNR (dB)')
plt.ylabel('Measured SNR post quantization (dB)')
plt.legend(bitwidthLeg)
plt.grid(True)
plt.ylim([0, max(trueSNR)+5])
plt.xlim([0, max(trueSNR)+5])
