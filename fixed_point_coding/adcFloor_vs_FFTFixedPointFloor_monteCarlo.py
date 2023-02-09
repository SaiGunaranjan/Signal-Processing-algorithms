# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 21:13:23 2022

@author: Sai Gunaranjan
"""


"""
Theoretical dynamic range set by a fixed point radix 2 normalized fft is given in the reference below:
Page 668 of the book Discrete-time signal processing by Oppenheim and Schafer(https://research.iaun.ac.ir/pd/naghsh/pdfs/UploadFile_2230.pdf), Eq 9.69
"""

import numpy as np
import matplotlib.pyplot as plt
from fixedPointLibrary import convert_float_to_fixedPointInt, dropFractionalBits_fixedPointInt,\
    convert_Complexfloat_to_fixedPointInt, fixedPointDFT, fixedPointfft_as_successive_dft, \
        fixedPointfft_as_successive_dft_scaled
import time as time



plt.close('all')

numTimeSamples = 2048
numRangeSamples = numTimeSamples//2
windowFunction = np.hanning(numTimeSamples)

IntBitsSignal = 1
FracBitsSignal = 15
signBitSignal = 1
numBits = IntBitsSignal + FracBitsSignal

IntBitsWindow = 1
FracBitsWindow = 15
signBitWindow = 1
windowFunctionFixedPoint = convert_float_to_fixedPointInt(windowFunction,IntBitsWindow,FracBitsWindow,signBitWindow) # 1Q31

""" ADC quantization/ signal fixed point noise floor is computed as delta**2/12
and is spread over N bins in the spectrum domain """
stepSizeDelta = 2/(2**numBits)
adcQuantizationNoiseIntegrated = (stepSizeDelta**2)/12
adcQuantizationNoisePerBin = adcQuantizationNoiseIntegrated/numTimeSamples
adcQuantizationNoisePerBindB = 10*np.log10(adcQuantizationNoisePerBin)
adcQuantizationNoisePerBindB = np.round(adcQuantizationNoisePerBindB)

theoreticalDynamicRange = 4*2**(-2*numBits)
theoreticalDynamicRangedB = 10*np.log10(theoreticalDynamicRange)
theoreticalDynamicRangedB = np.round(theoreticalDynamicRangedB)

targetRangeBins = 612 + np.random.uniform(-0.5,0.5,1)[0]
noiseFloorPerBinArray = np.arange(-180,-50,5)
numNoiseEvals = len(noiseFloorPerBinArray)
noisePowerFlSigFlFFT = np.zeros((numNoiseEvals),dtype=np.float32)
noisePowerFixSigFlFFT =  np.zeros((numNoiseEvals),dtype=np.float32)
noisePowerFixSigFixFFTwBG = np.zeros((numNoiseEvals),dtype=np.float32)
noisePowerFixSigFixFFTwoBG = np.zeros((numNoiseEvals),dtype=np.float32)
count = 0
for noiseFloorPerBin in noiseFloorPerBinArray:
    noiseFloorPerBindBFs = noiseFloorPerBin # change from -50 to -150 to get which blocks domainates the noise floor
    totalNoisePowerdBFs = noiseFloorPerBindBFs + 10*np.log10(numTimeSamples)
    totalNoisePower = 10**(totalNoisePowerdBFs/10)
    totalNoiseSigma = np.sqrt(totalNoisePower)

    objectSNR_perBin = np.abs(noiseFloorPerBindBFs) - 20 # (-20 dB is to back off the signal power from 0 dBFs)
    signalPowerdBFs = objectSNR_perBin + noiseFloorPerBindBFs
    signalPower = 10**(signalPowerdBFs/10)
    signalAmplitude = np.sqrt(signalPower)
    signalPhase = np.exp(1j*np.random.uniform(-np.pi, np.pi))
    signalphasor = signalAmplitude*signalPhase

    signal = signalphasor*np.exp(1j*2*np.pi*targetRangeBins*np.arange(numTimeSamples)/numTimeSamples)
    noise = (totalNoiseSigma/np.sqrt(2))*np.random.randn(numTimeSamples) + 1j*(totalNoiseSigma/np.sqrt(2))*np.random.randn(numTimeSamples)
    noisySignal = signal + noise

    signalWindowedFloat= noisySignal*windowFunction
    rfft = np.fft.fft(signalWindowedFloat)[0:numRangeSamples]/numTimeSamples
    rfftpsd = np.abs(rfft)**2
    rfftSpec = 10*np.log10(rfftpsd)
    noisePowerFlSigFlFFT[count] = 10*np.log10(np.mean(rfftpsd[0:500]))

    """ Fixed point implementation"""

    noisySignalFixedPoint = convert_Complexfloat_to_fixedPointInt(noisySignal,IntBitsSignal,FracBitsSignal,signBitSignal)
    signalWindowedFixedPoint = noisySignalFixedPoint*windowFunctionFixedPoint # 32 bit x 32 bit = 64 bit
    """
    Input signal swings from +/- 2**31. Window also allocated to +/-2**31.
    So multiplying these two results is a swing of +/- 2**32 * 2**31 = 2**62.
    Hence the input fractional bits = 62.
    Now, we want the final output post multiplication to be of swing +/-2**31 again.
    So the output fractional bits = 31
    We need to bring down 2**62 to 2**31. Hence we need to drop 62-31 = 31 bits.
    Since the output has a swing from +/-2**31, to get back the true float value, divide by 2**31

    """
    inputArrFracBits = FracBitsSignal + FracBitsWindow
    outputArrFracBits = FracBitsSignal
    signalWindowedRealBitsDropped = dropFractionalBits_fixedPointInt(np.real(signalWindowedFixedPoint).astype('int64'), inputArrFracBits, outputArrFracBits)
    signalWindowedImagBitsDropped = dropFractionalBits_fixedPointInt(np.imag(signalWindowedFixedPoint).astype('int64'), inputArrFracBits, outputArrFracBits)
    """ The signal passed to the above function should be of datatype int64. This is beacause,
    the signal is 32 bit, window is also 32 bit and hence the product will be 64 bits. So, if we type cast it as int32, there will be clipping/saturation
    to the max 32 bit signed value i.e. -2**31, 2**31-1
    """
    signalWindowedBitsDropped = signalWindowedRealBitsDropped + 1j*signalWindowedImagBitsDropped
    signalWindowedCovertFloat = signalWindowedBitsDropped/(2**outputArrFracBits)


    """ Numpy FFT/ Floating point FFT on ADC/fixed point signal"""
    adcSignalFFT = np.fft.fft(signalWindowedBitsDropped)[0:numRangeSamples]/(numTimeSamples*(2**FracBitsSignal))
    adcSignalpsd = np.abs(adcSignalFFT)**2
    adcSignalFFTSpec = 10*np.log10(adcSignalpsd)
    noisePowerFixSigFlFFT[count] = 10*np.log10(np.mean(adcSignalpsd[0:500]))


    """ Fixed Point Radix-2 FFT"""
    radix2fftFixedPoint = fixedPointfft_as_successive_dft(signalWindowedBitsDropped, IntBitsSignal, FracBitsSignal, signBitSignal)
    radix2fftFixedPoint = (radix2fftFixedPoint[0:numRangeSamples].squeeze())/numTimeSamples
    """ Take care of 0 values in the FFT """
    # radix2fftFixedPoint[(np.real(radix2fftFixedPoint) == 0) & (np.imag(radix2fftFixedPoint)==0)] = 1
    """ """
    radix2fftFixedPointConvertToFloat = radix2fftFixedPoint/(2**FracBitsSignal)
    radix2fftFixedPointConvertToFloatpsd = np.abs(radix2fftFixedPointConvertToFloat)**2
    radix2fftFixedPointConvertToFloatSpec = 10*np.log10(radix2fftFixedPointConvertToFloatpsd)
    noisePowerFixSigFixFFTwBG[count] = 10*np.log10(np.mean(radix2fftFixedPointConvertToFloatpsd[0:500]))


    """ Fixed Point Normalized Radix-2 FFT"""
    radix2fftNormalizedFixedPoint = fixedPointfft_as_successive_dft_scaled(signalWindowedBitsDropped, IntBitsSignal, FracBitsSignal, signBitSignal)
    radix2fftNormalizedFixedPoint = (radix2fftNormalizedFixedPoint[0:numRangeSamples].squeeze())
    """ Take care of 0 values in the FFT """
    radix2fftNormalizedFixedPoint[(np.real(radix2fftNormalizedFixedPoint) == 0) & (np.imag(radix2fftNormalizedFixedPoint)==0)] = 1
    """ """
    radix2fftNormalizedFixedPointConvertToFloat = radix2fftNormalizedFixedPoint/(2**FracBitsSignal)
    radix2fftNormalizedFixedPointConvertToFloatpsd = np.abs(radix2fftNormalizedFixedPointConvertToFloat)**2
    radix2fftNormalizedFixedPointConvertToFloatSpec = 10*np.log10(radix2fftNormalizedFixedPointConvertToFloatpsd)
    noisePowerFixSigFixFFTwoBG[count] = 10*np.log10(np.mean(radix2fftNormalizedFixedPointConvertToFloatpsd[0:500]))

    count += 1


plt.figure(1,figsize=(20,10))
plt.title('Processing blocks setting the noise floor')
plt.plot(noiseFloorPerBinArray,noisePowerFlSigFlFFT,lw=6,alpha=0.3,label='floating pt signal, floating pt fft')
plt.plot(noiseFloorPerBinArray,noisePowerFixSigFlFFT, lw=4,alpha=0.5,label='ADC signal, floating pt fft')
plt.plot(noiseFloorPerBinArray,noisePowerFixSigFixFFTwBG,lw=2, alpha=0.7, label='ADC signal, ' + str(numBits) + ' bit Fixed Point Radix-2 FFT with bit growth')
plt.plot(noiseFloorPerBinArray,noisePowerFixSigFixFFTwoBG,label='ADC signal, ' + str(numBits) + ' bit Fixed Point Radix-2 FFT without bit growth')
plt.axhline(adcQuantizationNoisePerBindB,ls='--',color='k',label='ADC quantization noise/bin with ' + str(numBits) + ' bits = ' + str(adcQuantizationNoisePerBindB) + ' dB')
plt.axhline(theoreticalDynamicRangedB,color='k',label='Theoret. DR with ' + str(numBits) + ' bits & fixed point radix-2 FFT without bit growth = ' + str(theoreticalDynamicRangedB) + ' dB')
plt.xlabel('programmed Noise floor (dBFs/bin)')
plt.ylabel('Measured Noise floor (dBFs/bin)')
plt.grid('True')
plt.legend()



plt.figure(2,figsize=(20,10))
plt.title('Range spectrum')
plt.plot(rfftSpec,lw=6,alpha=0.3,label='floating pt signal, floating pt fft')
plt.plot(adcSignalFFTSpec,lw=4,alpha=0.5,label='ADC signal, floating pt fft')
plt.plot(radix2fftFixedPointConvertToFloatSpec,lw=2, alpha=0.7, label='ADC signal, ' + str(numBits) + ' bit Fixed Point Radix-2 FFT with bit growth')
plt.plot(radix2fftNormalizedFixedPointConvertToFloatSpec,label='ADC signal, ' + str(numBits) + ' bit Fixed Point Radix-2 FFT without bit growth')
plt.axhline(adcQuantizationNoisePerBindB,ls='--',color='k',label='ADC quantization noise/bin with ' + str(numBits) + ' bits = ' + str(adcQuantizationNoisePerBindB) + ' dB')
plt.axhline(noiseFloorPerBindBFs,ls='-.',color='k',label='Programmed noise/bin = ' + str(noiseFloorPerBindBFs) + ' dB')
plt.axhline(theoreticalDynamicRangedB,color='k',label='Theoret. DR with ' + str(numBits) + ' bits & fixed point radix-2 FFT without bit growth = ' + str(theoreticalDynamicRangedB) + ' dB')
plt.xlabel('bins')
plt.grid('True')
plt.legend()





