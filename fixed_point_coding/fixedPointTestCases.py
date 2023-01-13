# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 21:13:23 2022

@author: Sai Gunaranjan
"""


import numpy as np
import matplotlib.pyplot as plt
from fixedPointLibrary import convert_float_to_fixedPointInt, dropFractionalBits_fixedPointInt,\
    convert_Complexfloat_to_fixedPointInt, matrixMultiplicationFixedPointComplexInput
import time as time




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
# dynamicRangedB = 20 # Eg. -10 dBsm and +10 dBsm RCS targets have an effective dynamic range of 20 dB
# referenceSNR = 20
# objectSNR_perBin = np.array([referenceSNR+dynamicRangedB])
objectSNR_perBin = np.array([40])
signalPowerdBFs = objectSNR_perBin + noiseFloorPerBindBFs
signalPower = 10**(signalPowerdBFs/10)
signalAmplitude = np.sqrt(signalPower)
signalPhase = np.exp(1j*np.random.uniform(-np.pi, np.pi))
signalphasor = signalAmplitude*signalPhase

targetRangeBins = 612 + np.random.uniform(-0.5,0.5,1)[0]
signal = signalphasor*np.exp(1j*2*np.pi*targetRangeBins*np.arange(numTimeSamples)/numTimeSamples)
noise = (totalNoiseSigma/np.sqrt(2))*np.random.randn(numTimeSamples) + 1j*(totalNoiseSigma/np.sqrt(2))*np.random.randn(numTimeSamples)
noisySignal = signal + noise

signalWindowedFloat= noisySignal*windowFunction
rfft = np.fft.fft(signalWindowedFloat)[0:numRangeSamples]/numTimeSamples

""" Fixed point implementation"""

IntBitsSignal = 1
FracBitsSignal = 31
signBitSignal = 1

noisySignalFixedPoint = convert_Complexfloat_to_fixedPointInt(noisySignal,IntBitsSignal,FracBitsSignal,signBitSignal)

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


"""
Fourier transform using matrix based DFT

"""
timeInd = np.arange(numTimeSamples)
numFFTBins = numTimeSamples # For now assume same number of FFT bins as number of samples.
freqInd = np.arange(numFFTBins)
#Also, assume for now that the signal length is always a power of 2. This is because, for fixed point integer operations, it is easy to work with scale factors which are powers of 2
DFTMatrix = np.exp(-1j*2*np.pi*freqInd[:,None]*timeInd[None,:]/numFFTBins)

rfft_dftMethod = (DFTMatrix @ signalWindowedFloat)/numTimeSamples
rfft_dftMethod = rfft_dftMethod[0:numRangeSamples]


numIntBits = 1
numFracBits = 31
numSignBits = 1
DFTMatrixFixedPoint = convert_Complexfloat_to_fixedPointInt(DFTMatrix, numIntBits, numFracBits, numSignBits)

inputArrFracBits = 31
outputArrFracBits = 31
t1 = time.time()
rfft_dftFixedPoint = matrixMultiplicationFixedPointComplexInput(DFTMatrixFixedPoint, signalWindowedBitsDropped[:,None], inputArrFracBits, outputArrFracBits)
t2 = time.time()
rfft_dftFixedPoint = (rfft_dftFixedPoint[0:numRangeSamples].squeeze())/numTimeSamples
rfft_dftFixedPointConvertToFloat = rfft_dftFixedPoint/(2**outputArrFracBits)

timeForFixedPointDFT = t2-t1
print('Total time for Fixed point DFT = {0:.2f} sec'.format(timeForFixedPointDFT))

rfftSpec = 20*np.log10(np.abs(rfft))
rfftSpec -= np.amax(rfftSpec)

rfft_dftMethodSpec = 20*np.log10(np.abs(rfft_dftMethod))
rfft_dftMethodSpec -= np.amax(rfft_dftMethodSpec)

rfft_dftFixedPointConvertToFloatSpec = 20*np.log10(np.abs(rfft_dftFixedPointConvertToFloat))
rfft_dftFixedPointConvertToFloatSpec -= np.amax(rfft_dftFixedPointConvertToFloatSpec)

plt.figure(2,figsize=(20,10))
plt.title('Range spectrum')
plt.plot(rfftSpec,label='Spectrum using FFT')
plt.plot(rfft_dftMethodSpec,alpha=0.5,label='Spectrum using DFT')
plt.plot(rfft_dftFixedPointConvertToFloatSpec,label='Spectrum using ' + str(outputArrFracBits) + ' bit Fixed Point DFT')
plt.xlabel('bins')
plt.grid('True')
plt.legend()




# plt.figure(1,figsize=(20,10))
# plt.title('Range Spectrum')
# plt.plot(20*np.log10(np.abs(rfft)))
# plt.grid(True)