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
numDopplerSamples = 64
numAngleSamples = 4

noiseFloorPerBindBFs = -150 # change from -50 to -150 to get which blocks domainates the noise floor
totalNoisePowerdBFs = noiseFloorPerBindBFs + 10*np.log10(numTimeSamples)
totalNoisePower = 10**(totalNoisePowerdBFs/10)
totalNoiseSigma = np.sqrt(totalNoisePower)



"""
Dynamic range test post Range FFT
"""
windowFunction = np.hanning(numTimeSamples)
# dynamicRangedB = 20 # Eg. -10 dBsm and +10 dBsm RCS targets have an effective dynamic range of 20 dB
objectSNR_perBin = np.abs(noiseFloorPerBindBFs) - 20 # (-20 dB is to back off the signal power from 0 dBFs)
signalPowerdBFs = objectSNR_perBin + noiseFloorPerBindBFs
signalPower = 10**(signalPowerdBFs/10)
signalAmplitude = np.sqrt(signalPower)
signalPhase = np.exp(1j*np.random.uniform(-np.pi, np.pi))
signalphasor = signalAmplitude*signalPhase

targetRangeBins = 612 + np.random.uniform(-0.5,0.5,1)[0]
signal = signalphasor*np.exp(1j*2*np.pi*targetRangeBins*np.arange(numTimeSamples)/numTimeSamples)
# signal = np.exp(1j*2*np.pi*targetRangeBins*np.arange(numTimeSamples)/numTimeSamples)
noise = (totalNoiseSigma/np.sqrt(2))*np.random.randn(numTimeSamples) + 1j*(totalNoiseSigma/np.sqrt(2))*np.random.randn(numTimeSamples)
noisySignal = signal + noise

signalWindowedFloat= noisySignal*windowFunction
rfft = np.fft.fft(signalWindowedFloat)[0:numRangeSamples]/numTimeSamples
rfftSpec = 20*np.log10(np.abs(rfft))
# rfftSpec -= np.amax(rfftSpec)


""" Fixed point implementation"""

IntBitsSignal = 1
FracBitsSignal = 15 # 31
signBitSignal = 1
numBits = IntBitsSignal + FracBitsSignal

noisySignalFixedPoint = convert_Complexfloat_to_fixedPointInt(noisySignal,IntBitsSignal,FracBitsSignal,signBitSignal)

IntBitsWindow = 1
FracBitsWindow = 15 #31
signBitWindow = 1
windowFunctionFixedPoint = convert_float_to_fixedPointInt(windowFunction,IntBitsWindow,FracBitsWindow,signBitWindow) # 1Q31

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
inputArrFracBits = FracBitsSignal + FracBitsWindow #30 # 62
outputArrFracBits = FracBitsSignal #15 # 31
signalWindowedRealBitsDropped = dropFractionalBits_fixedPointInt(np.real(signalWindowedFixedPoint).astype('int64'), inputArrFracBits, outputArrFracBits)
signalWindowedImagBitsDropped = dropFractionalBits_fixedPointInt(np.imag(signalWindowedFixedPoint).astype('int64'), inputArrFracBits, outputArrFracBits)
""" The signal passed to the above function should be of datatype int64. This is beacause,
the signal is 32 bit, window is also 32 bit and hence the product will be 64 bits. So, if we type cast it as int32, there will be clipping/saturation
to the max 32 bit signed value i.e. -2**31, 2**31-1
"""
signalWindowedBitsDropped = signalWindowedRealBitsDropped + 1j*signalWindowedImagBitsDropped
signalWindowedCovertFloat = signalWindowedBitsDropped/(2**outputArrFracBits)


""" Disable DFT since compute time is high and is practically not implementable"""
if 0:

    """ Fourier transform using matrix based DFT """
    timeInd = np.arange(numTimeSamples)
    numFFTBins = numTimeSamples # For now assume same number of FFT bins as number of samples.
    freqInd = np.arange(numFFTBins)
    #Also, assume for now that the signal length is always a power of 2. This is because, for fixed point integer operations, it is easy to work with scale factors which are powers of 2
    DFTMatrix = np.exp(-1j*2*np.pi*freqInd[:,None]*timeInd[None,:]/numFFTBins)
    rfft_dftMethod = (DFTMatrix @ signalWindowedFloat)/numTimeSamples
    rfft_dftMethod = rfft_dftMethod[0:numRangeSamples]
    rfft_dftMethodSpec = 20*np.log10(np.abs(rfft_dftMethod))
    # rfft_dftMethodSpec -= np.amax(rfft_dftMethodSpec)

    """ Fixed point DFT """
    inputArrFracBits = FracBitsSignal#15 #31
    outputArrFracBits = FracBitsSignal#15 #31
    t1 = time.time()
    rfft_dftFixedPoint = fixedPointDFT(signalWindowedBitsDropped, numFFTBins, inputArrFracBits, outputArrFracBits)
    t2 = time.time()
    rfft_dftFixedPoint = (rfft_dftFixedPoint[0:numRangeSamples].squeeze())/numTimeSamples
    rfft_dftFixedPointConvertToFloat = rfft_dftFixedPoint/(2**outputArrFracBits)
    rfft_dftFixedPointConvertToFloatSpec = 20*np.log10(np.abs(rfft_dftFixedPointConvertToFloat))
    # rfft_dftFixedPointConvertToFloatSpec -= np.amax(rfft_dftFixedPointConvertToFloatSpec)
    timeForFixedPointDFT = t2-t1
    print('Total time for Fixed point DFT = {0:.2f} sec'.format(timeForFixedPointDFT))


""" Numpy FFT/ Floating point FFT on ADC/fixed point signal"""
adcSignalFFT = np.fft.fft(signalWindowedBitsDropped)[0:numRangeSamples]/(numTimeSamples*(2**FracBitsSignal))
adcSignalFFTSpec = 20*np.log10(np.abs(adcSignalFFT))
# adcSignalFFTSpec -= np.amax(adcSignalFFTSpec)

""" ADC quantization/ signal fixed point noise floor is computed as delta**2/12
and is spread over N bins in the spectrum domain """
stepSizeDelta = 2/(2**numBits)
adcQuantizationNoiseIntegrated = (stepSizeDelta**2)/12
adcQuantizationNoisePerBin = adcQuantizationNoiseIntegrated/numTimeSamples
adcQuantizationNoisePerBindB = 10*np.log10(adcQuantizationNoisePerBin)
adcQuantizationNoisePerBindB = np.round(adcQuantizationNoisePerBindB)


""" Fixed Point Radix-2 FFT"""
radix2fftFixedPoint = fixedPointfft_as_successive_dft(signalWindowedBitsDropped, IntBitsSignal, FracBitsSignal, signBitSignal)
radix2fftFixedPoint = (radix2fftFixedPoint[0:numRangeSamples].squeeze())/numTimeSamples
radix2fftFixedPointConvertToFloat = radix2fftFixedPoint/(2**FracBitsSignal)
radix2fftFixedPointConvertToFloatSpec = 20*np.log10(np.abs(radix2fftFixedPointConvertToFloat))
# radix2fftFixedPointConvertToFloatSpec -= np.amax(radix2fftFixedPointConvertToFloatSpec)

""" Fixed Point Normalized Radix-2 FFT"""
radix2fftNormalizedFixedPoint = fixedPointfft_as_successive_dft_scaled(signalWindowedBitsDropped, IntBitsSignal, FracBitsSignal, signBitSignal)
radix2fftNormalizedFixedPoint = (radix2fftNormalizedFixedPoint[0:numRangeSamples].squeeze())
""" Take care of 0 values in the FFT """
radix2fftNormalizedFixedPoint[(np.real(radix2fftNormalizedFixedPoint) == 0) & (np.imag(radix2fftNormalizedFixedPoint)==0)] = 1
""" """
radix2fftNormalizedFixedPointConvertToFloat = radix2fftNormalizedFixedPoint/(2**FracBitsSignal)
radix2fftNormalizedFixedPointConvertToFloatSpec = 20*np.log10(np.abs(radix2fftNormalizedFixedPointConvertToFloat))


theoreticalDynamicRange = 4*2**(-2*numBits)
theoreticalDynamicRangedB = 10*np.log10(theoreticalDynamicRange)
theoreticalDynamicRangedB = np.round(theoreticalDynamicRangedB)


plt.figure(2,figsize=(20,10))
plt.title('Range spectrum')
plt.plot(rfftSpec,lw=6,alpha=0.3,label='floating pt signal, floating pt fft')
plt.plot(adcSignalFFTSpec,lw=4,alpha=0.5,label='ADC signal, floating pt fft')
# plt.plot(rfft_dftMethodSpec,lw=4,alpha=0.5,label='Spectrum using DFT')
# plt.plot(rfft_dftFixedPointConvertToFloatSpec,label='Spectrum using ' + str(outputArrFracBits) + ' bit Fixed Point DFT')
plt.plot(radix2fftFixedPointConvertToFloatSpec,lw=2, alpha=0.7, label='ADC signal, ' + str(numBits) + ' bit Fixed Point Radix-2 FFT with bit growth')
plt.plot(radix2fftNormalizedFixedPointConvertToFloatSpec,label='ADC signal, ' + str(numBits) + ' bit Fixed Point Radix-2 FFT without bit growth')
plt.axhline(adcQuantizationNoisePerBindB,ls='--',color='k',label='ADC quantization noise/bin with ' + str(numBits) + ' bits = ' + str(adcQuantizationNoisePerBindB) + ' dB')
plt.axhline(noiseFloorPerBindBFs,ls='-.',color='k',label='Programmed noise/bin = ' + str(noiseFloorPerBindBFs) + ' dB')
plt.axhline(theoreticalDynamicRangedB,color='k',label='Theoret. DR with ' + str(numBits) + ' bits & fixed point radix-2 FFT without bit growth = ' + str(theoreticalDynamicRangedB) + ' dB')
plt.xlabel('bins')
plt.grid('True')
plt.legend()




if 0:
    plt.figure(1,figsize=(20,10))
    plt.suptitle('Windowed Signal: Floating vs Fixed')
    plt.subplot(2,2,1)
    plt.title('Real part')
    plt.plot(np.real(signalWindowedFloat[numTimeSamples//2 - 50:numTimeSamples//2 + 50]),label='Floating Point')
    plt.plot(np.real(signalWindowedCovertFloat[numTimeSamples//2 - 50:numTimeSamples//2 + 50]),label='Fixed Point convert to Float')
    plt.grid(True)
    plt.legend()

    plt.subplot(2,2,2)
    plt.title('Imag part')
    plt.plot(np.imag(signalWindowedFloat[numTimeSamples//2 - 50:numTimeSamples//2 + 50]),label='Floating Point')
    plt.plot(np.imag(signalWindowedCovertFloat[numTimeSamples//2 - 50:numTimeSamples//2 + 50]),label='Fixed Point convert to Float')
    plt.grid(True)
    plt.legend()

    plt.subplot(2,2,3)
    plt.title('Real part error')
    plt.plot(np.real(signalWindowedFloat) - np.real(signalWindowedCovertFloat))
    plt.grid(True)

    plt.subplot(2,2,4)
    plt.title('Imag part error')
    plt.plot(np.imag(signalWindowedFloat) - np.imag(signalWindowedCovertFloat))
    plt.grid(True)
