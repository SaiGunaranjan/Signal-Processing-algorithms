# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 21:13:23 2022

@author: Sai Gunaranjan
"""


import numpy as np
import matplotlib.pyplot as plt


def convert_float_to_fixedPointInt(floatArr, numIntBits, numFracBits, numSignBits):

    """
    floatArr: must be a real valued array
    """

    effectiveNumBits = numIntBits + numFracBits - numSignBits # Number of integer bits also includes the signed bit

    posmaxVal = 2**effectiveNumBits - 1

    scaledValue = np.floor(floatArr*(2**numFracBits) + 0.5)

    scaledValue[scaledValue>posmaxVal] = posmaxVal # clip/saturate values above posmaxVal to posmaxVal to avoid overflow

    if numSignBits:
        negativeminVal = -(2**effectiveNumBits) # If signed
        scaledValue[scaledValue < negativeminVal] = negativeminVal # For signed integers, clip/saturate values below negativeminVal to negativeminVal to avoid underflow
    else:
        scaledValue[scaledValue < 0] = 0 # For unsigned integers, clip/saturate values below 0 to 0 to avoid underflow

    return scaledValue.astype('int32') # can also be type cast as float 32. In that case, it will be an integer but with a .0


# def convert_Complexfloat_to_fixedPointInt(complexArr, numIntBits, numFracBits, numSignBits):

#     """
#     complexArr: must be a complex valued array
#     """

#     effectiveNumBits = numIntBits + numFracBits - numSignBits # Number of integer bits also includes the signed bit

#     posmaxVal = 2**effectiveNumBits - 1
#     negativeminVal = -(2**effectiveNumBits)

#     realPart = np.real(complexArr)
#     imagPart = np.imag(complexArr)

#     scaledValueRealPart = np.floor(realPart*(2**numFracBits) + 0.5)
#     scaledValueImagPart = np.floor(imagPart*(2**numFracBits) + 0.5)

#     scaledValueRealPart[scaledValueRealPart>posmaxVal] = posmaxVal # clip/saturate values above posmaxVal to posmaxVal to avoid overflow
#     scaledValueImagPart[scaledValueImagPart>posmaxVal] = posmaxVal # clip/saturate values above posmaxVal to posmaxVal to avoid overflow

#     if numSignBits:
#         scaledValueRealPart[scaledValueRealPart < negativeminVal] = negativeminVal # For signed integers, clip/saturate values below negativeminVal to negativeminVal to avoid underflow
#         scaledValueImagPart[scaledValueImagPart < negativeminVal] = negativeminVal # For signed integers, clip/saturate values below negativeminVal to negativeminVal to avoid underflow
#     else:
#         scaledValueRealPart[scaledValueRealPart < 0] = 0 # For unsigned integers, clip/saturate values below 0 to 0 to avoid underflow
#         scaledValueImagPart[scaledValueImagPart < 0] = 0 # For unsigned integers, clip/saturate values below 0 to 0 to avoid underflow

#     scaledcomplexArr = (scaledValueRealPart + 1j*scaledValueImagPart).astype('complex64') # can also be type cast as float 32. In that case, it will be an integer but with a .0

#     return scaledcomplexArr


def dropFractionalBits_fixedPointInt(inputFixedPointArr, inputArrFracBits, outputArrFracBits):

    """
    inputFixedPointArr : should be an array of integers in int32 format
    """

    numFracBitsToBeDropped = inputArrFracBits - outputArrFracBits # Works only when inputArrFracBits >= outputArrFracBits

    # outputFixedPointArr = (inputFixedPointArr + 2**(numFracBitsToBeDropped-1)) >> numFracBitsToBeDropped
    outputFixedPointArr = (inputFixedPointArr + (1<<(numFracBitsToBeDropped-1))) >> numFracBitsToBeDropped # Replaced 2**(n-1) with bit shift operation of shifting binary 1 by n-1 bits

    return outputFixedPointArr


def addFractionalBits_fixedPointInt(inputFixedPointArr, inputArrFracBits, outputArrFracBits):

    """
    inputFixedPointArr : should be an array of integers in int32 format
    """

    numFracBitsToBeAdded = outputArrFracBits - inputArrFracBits # Works only when inputArrFracBits >= outputArrFracBits

    outputFixedPointArr = inputFixedPointArr << numFracBitsToBeAdded

    return outputFixedPointArr



plt.close('all')

numTimeSamples = 2048
numRangeSamples = numTimeSamples//2
numDopplerSamples = 64
numAngleSamples = 4

noiseFloorPerBindBFs = -90
totalNoisePowerdBFs = noiseFloorPerBindBFs + 10*np.log10(numTimeSamples)
totalNoisePower = 10**(totalNoisePowerdBFs/10)
totalNoiseSigma = np.sqrt(totalNoisePower)



""" Case 1 : Dynamic range test post Range FFT
"""
windowFunction = np.hanning(numTimeSamples)
dynamicRangedB = 20 # Eg. -10 dBsm and +10 dBsm Rcs targets have an effective dynamic range of 20 dB
referenceSNR = 20
objectSNR_perBin = np.array([referenceSNR+dynamicRangedB])
signalPowerdBFs = objectSNR_perBin + noiseFloorPerBindBFs
signalPower = 10**(signalPowerdBFs/10)
signalAmplitude = np.sqrt(signalPower)
signalPhase = np.exp(1j*np.random.uniform(-np.pi, np.pi))
signalphasor = signalAmplitude*signalPhase

targetRangeBins = 612 + 0*np.random.uniform(-0.5,0.5,1)[0]
signal = np.exp(1j*2*np.pi*targetRangeBins*np.arange(numTimeSamples)/numTimeSamples)
noise = (totalNoiseSigma/np.sqrt(2))*np.random.randn(numTimeSamples) + 1j*(totalNoiseSigma/np.sqrt(2))*np.random.randn(numTimeSamples)
noisySignal = signal + noise

signalWindowedFloat= noisySignal*windowFunction
rfft = np.fft.fft(signalWindowedFloat)[0:numRangeSamples]/numTimeSamples

""" Fixed point implementation"""

IntBitsSignal = 1
FracBitsSignal = 31
signBitSignal = 1
noisySignalFixedPointReal = convert_float_to_fixedPointInt(np.real(noisySignal),IntBitsSignal,FracBitsSignal,signBitSignal)
noisySignalFixedPointImag = convert_float_to_fixedPointInt(np.imag(noisySignal),IntBitsSignal,FracBitsSignal,signBitSignal)
noisySignalFixedPoint = noisySignalFixedPointReal + 1j*noisySignalFixedPointImag # 1Q31

IntBitsWindow = 1
FracBitsWindow = 31
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
inputArrFracBits = 62
outputArrFracBits = 31
signalWindowedRealBitsDropped = dropFractionalBits_fixedPointInt(np.real(signalWindowedFixedPoint).astype('int64'), inputArrFracBits, outputArrFracBits)
signalWindowedImagBitsDropped = dropFractionalBits_fixedPointInt(np.imag(signalWindowedFixedPoint).astype('int64'), inputArrFracBits, outputArrFracBits)
""" The signal passed to the above function should be of datatype int64. This is beacause,
the signal is 32 bit, window is also 32 bit and hence the product will be 64 bits. So, if we type cast it as int32, there will be clipping/saturation
to the max 32 bit signed value i.e. -2**31, 2**31-1
"""

signalWindowedBitsDropped = signalWindowedRealBitsDropped + 1j*signalWindowedImagBitsDropped
signalWindowedCovertFloat = signalWindowedBitsDropped/(2**outputArrFracBits)

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


# """
# Fourier transform using matrix based DFT

# """
# timeInd = np.arange(numTimeSamples)
# numFFTBins = numTimeSamples # For now assume same number of FFT bins as number of samples.
# freqInd = np.arange(numFFTBins)
# #Also, assume for now that the signal length is always a power of 2. This is because, for fixed point integer operations, it is easy to work with scale factors which are powers of 2
# DFTMatrix = np.exp(-1j*2*np.pi*freqInd[:,None]*timeInd[None,:]/numFFTBins)




# plt.figure(1,figsize=(20,10))
# plt.title('Range Spectrum')
# plt.plot(20*np.log10(np.abs(rfft)))
# plt.grid(True)