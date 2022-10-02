# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 10:29:28 2022

@author: Sai Gunaranjan
"""


import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

numSamp = 128
numFFTBins = 128
freqBin = 6 + 0*np.random.uniform(0,1) # np.random.randint(0,numSamp) + np.random.uniform(0,1)
if (freqBin >= numFFTBins//2):
    freqBin -= numFFTBins
fftAxis = np.arange(-numFFTBins//2, numFFTBins//2)
eps = 1e-10

phaseMismatchFactorDeg = 0#np.random.uniform(0,80) #10
phaseMismatchFactorRad = (phaseMismatchFactorDeg/180) * np.pi

amplitudeMismatchFactor = np.arange(0.03,1,0.03)#np.random.uniform(0.03,0.5) #0.3
""" I channel and Q channel signals at the receiver chain due to receiver chain having an IQ imbalance i.e
I channel and Q channel and not exactly orthogonal to each other and have
amplitude and phase mismatches"""

for alpha in amplitudeMismatchFactor:
    iChannelSignalWithIQMM = alpha*np.cos(2*np.pi*freqBin*np.arange(numSamp)/numSamp)
    qChannelSignalWithIQMM = np.sin(2*np.pi*freqBin*np.arange(numSamp)/numSamp + phaseMismatchFactorRad)

    complexBaseBandSignalWithIQMM = iChannelSignalWithIQMM + 1j*qChannelSignalWithIQMM
    complexBaseBandSignalWithIQMMFFT = np.fft.fft(complexBaseBandSignalWithIQMM)/numSamp
    complexBaseBandSignalWithIQMMFFT = np.fft.fftshift(complexBaseBandSignalWithIQMMFFT)


    plt.figure(1,figsize=(20,10),dpi=200)
    plt.clf()
    plt.suptitle('IQMM with Amplitude mismatch = ' + str(np.round(alpha,2)))
    plt.subplot(1,2,1)
    plt.title('Time domain')
    plt.plot(iChannelSignalWithIQMM,label='I channel with IQMM')
    plt.plot(qChannelSignalWithIQMM,label='Q channel with IQMM')
    plt.xlabel('signal index')
    plt.grid(True)
    plt.legend()
    plt.ylim([-1,1])

    plt.subplot(1,2,2)
    plt.title('Frequency domain')
    # plt.plot(fftAxis, 10*np.log10(np.abs(complexBaseBandSignalWithIQMMFFT)**2 + eps),label='IQMM signal spectrum')
    # plt.ylim([-60,0])
    plt.plot(fftAxis, np.abs(complexBaseBandSignalWithIQMMFFT)**2,label='IQMM signal spectrum')
    plt.ylim([-0.2,1])
    plt.axvline(freqBin,label='True freq bin',color='k',alpha=0.4)
    plt.axvline(-freqBin,label='Image freq bin',color='k',alpha=0.4,ls='--')
    plt.xlabel('Frequency bin')
    plt.grid(True)
    plt.legend()


    plt.pause(0.05)

