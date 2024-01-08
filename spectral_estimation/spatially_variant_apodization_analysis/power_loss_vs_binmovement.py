# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 13:19:32 2023

@author: Sai Gunaranjan
"""




import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..//')
from spectral_estimation_lib import spatially_variant_apodization_multidimension

plt.close('all')

N = 2048
intbin = 1024
fractionalBin = np.arange(0,0.55,0.05)
binNum = intbin+fractionalBin
osrFact = 1
Nfft = N*osrFact

signal = np.exp(1j*2*np.pi*binNum[:,None]*np.arange(N)[None,:]/N)

signalFFT_rect = np.fft.fft(signal,n=Nfft,axis=1)/N
spectrumRect = 20*np.log10(np.abs(signalFFT_rect))
peakPowRect = np.amax(spectrumRect,axis=1)

signalFFT_hann = np.fft.fft(signal*np.hanning(N)[None,:],n=Nfft,axis=1)/N
spectrumHann = 20*np.log10(np.abs(signalFFT_hann))
peakPowHann = np.amax(spectrumHann,axis=1)

apod_axis = 1
signalApod, _ = spatially_variant_apodization_multidimension(signal, osrFact, apod_axis)
signalApod = np.fft.fftshift(signalApod,axes=(1,))
spectrumApod = 20*np.log10(np.abs(signalApod))
peakPowApod = np.amax(spectrumApod,axis=1)

plt.figure(1,figsize=(20,10),dpi=200)
plt.title('Power loss vs bin movement')
plt.plot(fractionalBin,peakPowRect,label='Rectangular window',lw=4,alpha=0.5)
plt.plot(fractionalBin,peakPowHann,label='Hanning window')
plt.plot(fractionalBin,peakPowApod,label='Apodization',color='black')
plt.grid(True)
plt.xlabel('Fractional bin value')
plt.ylabel('Signal power (dB)')
plt.legend()




