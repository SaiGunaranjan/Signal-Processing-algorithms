# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 23:43:07 2023

@author: Sai Gunaranjan
"""


import numpy as np
import matplotlib.pyplot as plt

def fft_as_successive_dft(signal):

    N = len(signal)

    if (N == 1):
        return signal
    else:
        evenIndexSignal = signal[0::2]
        oddIndexSignal = signal[1::2]
        evenIndexSignalFFT = fft_as_successive_dft(evenIndexSignal)
        oddIndexSignalFFT = fft_as_successive_dft(oddIndexSignal)

        # N = 2*len(evenIndexSignalFFT)
        k = np.arange(0,N/2)

        twiddleFactor = np.exp(-1j*2*np.pi*k/N)

        oddIndexSignalFFTTwiddleScaling = oddIndexSignalFFT*twiddleFactor
        fftTopHalf = evenIndexSignalFFT + oddIndexSignalFFTTwiddleScaling
        fftBottomHalf = evenIndexSignalFFT - oddIndexSignalFFTTwiddleScaling

        signalFFT = np.hstack((fftTopHalf,fftBottomHalf))

        return signalFFT

def fft_as_successive_dft_scaled(signal):

    N = len(signal)

    if (N == 1):
        return signal
    else:
        evenIndexSignal = signal[0::2]
        oddIndexSignal = signal[1::2]
        evenIndexSignalFFT = fft_as_successive_dft_scaled(evenIndexSignal)
        oddIndexSignalFFT = fft_as_successive_dft_scaled(oddIndexSignal)

        # N = 2*len(evenIndexSignalFFT)
        k = np.arange(0,N/2)

        twiddleFactor = np.exp(-1j*2*np.pi*k/N)

        oddIndexSignalFFTTwiddleScaling = oddIndexSignalFFT*twiddleFactor
        fftTopHalf = evenIndexSignalFFT + oddIndexSignalFFTTwiddleScaling
        fftBottomHalf = evenIndexSignalFFT - oddIndexSignalFFTTwiddleScaling

        signalFFT = np.hstack((fftTopHalf,fftBottomHalf))/2 # By 2 to scale the ouput at each stage.
        # Since there are log2(N) stages, the scaling becomes 2**(log2(N)) = N

        return signalFFT




def DFT(signal):

    numSamples = len(signal)
    dftMatrix = np.exp(-1j*2*np.pi*np.arange(numSamples)[:,None]*np.arange(numSamples)[None,:]/numSamples)
    signaldft = dftMatrix @ signal

    return signaldft

plt.close('all')
numSamples = 512
signalFreqBin = np.random.randint(0,numSamples)
signal = np.exp(1j*2*np.pi*signalFreqBin*np.arange(numSamples)/numSamples)

signalfft1 = np.fft.fft(signal)
signalfft2 = fft_as_successive_dft(signal)
signalfft3 = fft_as_successive_dft_scaled(signal)
signaldft = DFT(signal)

""" Timing the modules
Execute the following command on the terminal to get the timings for different implementations of the fourier transform

%timeit DFT(signal)
%timeit fft_as_successive_dft(signal)
%timeit np.fft.fft(signal)

"""


plt.figure(1,figsize=(20,10),dpi=200)
plt.plot(20*np.log10(np.abs(signalfft1)),label='numpy fft')
# plt.plot(20*np.log10(np.abs(signaldft)),label='DFT')
plt.plot(20*np.log10(np.abs(signalfft2)),lw=2,alpha=0.5,label='FFT as successive DFT')
plt.plot(20*np.log10(np.abs(signalfft3)),alpha=0.5,label='FFT as successive DFT scaled')
plt.grid(True)
plt.legend()