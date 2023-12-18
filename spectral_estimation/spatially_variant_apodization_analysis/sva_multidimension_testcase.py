# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 20:01:56 2020

@author: saiguna
"""


import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..//')
from spectral_estimation_lib import spatially_variant_apodization_multidimension


plt.close('all')

numADCSamp = 2048
numRamps = 512
rbin = 133
dbin = 150
adcsig = np.exp(1j*2*np.pi*rbin*np.arange(numADCSamp)/numADCSamp)
doppsig = np.exp(1j*2*np.pi*dbin*np.arange(numRamps)/numRamps)
radar_signal = adcsig[:,None] * doppsig[None,:]
radar_signal_range_fft = np.fft.fft(radar_signal,axis=0)/numADCSamp
radar_signal_range_fft_dopp_fft = np.fft.fft(radar_signal_range_fft,axis=1)/numRamps

""" SVA Optimized"""
apod_axis = 0
osrFact = 1
radar_signal_range_fftshifted,_ = spatially_variant_apodization_multidimension(radar_signal, osrFact, apod_axis)
radar_signal_range_apod = np.fft.fftshift(radar_signal_range_fftshifted,axes=(apod_axis,))

apod_axis = 1
osrFact = 1
radar_signal_range_apod_dopp_apod_fftshifted,_ = spatially_variant_apodization_multidimension(radar_signal_range_apod, osrFact, apod_axis)
radar_signal_range_apod_dopp_apod = np.fft.fftshift(radar_signal_range_apod_dopp_apod_fftshifted,axes=(apod_axis,))

plt.figure(1,figsize=(20,10),dpi=200)
plt.suptitle('FFT vs Apod')
plt.subplot(1,2,1)
plt.title('Range spectrum')
plt.plot(20*np.log10(np.abs(radar_signal_range_fft[:,100])),label='range FFT')
plt.plot(20*np.log10(np.abs(radar_signal_range_apod[:,100])),label='range Apod')
plt.grid(True)
plt.legend()
plt.axvline(rbin,ls='--',color='black')

plt.subplot(1,2,2)
plt.title('Doppler spectrum')
plt.plot(20*np.log10(np.abs(radar_signal_range_fft_dopp_fft[rbin,:])),label='Doppler FFT')
plt.plot(20*np.log10(np.abs(radar_signal_range_apod_dopp_apod[rbin,:])),label='Doppler Apod')
plt.grid(True)
plt.legend()
plt.axvline(dbin,ls='--',color='black')


