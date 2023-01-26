# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 23:43:07 2023

@author: Sai Gunaranjan
"""

import numpy as np
import matplotlib.pyplot as plt

def fft_as_successive_dft(signal):

    if (len(signal) == 1):
        return signal
    else:
        evenIndexSignal = signal[0::2]
        oddIndexSignal = signal[1::2]
        evenIndexSignalFFT = fft_as_successive_dft(evenIndexSignal)
        oddIndexSignalFFT = fft_as_successive_dft(oddIndexSignal)

        N = 2*len(evenIndexSignalFFT)
        k = np.arange(0,N/2)

        twiddleFactor = np.exp(-1j*2*np.pi*k/N)

        oddIndexSignalFFTTwiddleScaling = oddIndexSignalFFT*twiddleFactor
        fftTopHalf = evenIndexSignalFFT + oddIndexSignalFFTTwiddleScaling
        fftBottomHalf = evenIndexSignalFFT - oddIndexSignalFFTTwiddleScaling

        signalFFT = np.hstack((fftTopHalf,fftBottomHalf))

        return signalFFT


def DFT(signal):

    numSamples = len(signal)
    dftMatrix = np.exp(-1j*2*np.pi*np.arange(numSamples)[:,None]*np.arange(numSamples)[None,:]/numSamples)
    signaldft = dftMatrix @ signal

    return signaldft

numSamples = 512
signalFreqBin = np.random.randint(0,numSamples)
signal = np.exp(1j*2*np.pi*signalFreqBin*np.arange(numSamples)/numSamples)

signalfft1 = np.fft.fft(signal)
signalfft2 = fft_as_successive_dft(signal)
signaldft = DFT(signal)

""" Timing the modules
Execute the following command on the terminal
%timeit DFT(signal)
%timeit fft_as_successive_dft(signal)
%timeit np.fft.fft(signal)

"""


plt.figure(1,figsize=(20,10),dpi=200)
plt.plot(20*np.log10(np.abs(signalfft1)),label='numpy fft')
plt.plot(20*np.log10(np.abs(signaldft)),label='DFT')
plt.plot(20*np.log10(np.abs(signalfft2)),lw=2,alpha=0.5,label='FFT as successive DFT')
plt.grid(True)
plt.legend()