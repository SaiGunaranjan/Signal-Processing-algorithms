# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 18:30:18 2023

@author: Sai Gunaranjan
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..//')
from spectral_estimation_lib import spatially_variant_apodization_bruteforce, spatially_variant_apodization_optimized, iaa_recursive
from spectral_estimation_lib import music_backward as music



np.random.seed(10)

plt.close('all')
num_samples = 32
c = 3e8
fc = 79e9
lamda = c/fc
mimoSpacing = lamda/2
fsSpatial = lamda/mimoSpacing
nativeAngResDeg = np.arcsin(fsSpatial/num_samples)*180/np.pi
print('Native Angular Resolution = {0:.2f} deg'.format(nativeAngResDeg))
num_sources = 2

## RF parameters
thermalNoise = -174 # dBm/Hz
noiseFigure = 10 # dB
baseBandgain = 34 #dB
adcSamplingRate = 56.25e6 # 56.25 MHz
adcSamplingTime = 1/adcSamplingRate # s
dBFs_to_dBm = 10
object_snr = np.array([40,35])
totalNoisePower_dBm = thermalNoise + noiseFigure + baseBandgain + 10*np.log10(adcSamplingRate)
totalNoisePower_dBFs = totalNoisePower_dBm - 10
noiseFloor_perBin = -90#totalNoisePower_dBFs - 10*np.log10(num_samples) # dBFs/bin
noisePower_perBin = 10**(noiseFloor_perBin/10)
totalNoisePower = noisePower_perBin*num_samples # sigmasquare totalNoisePower
noise_sigma = np.sqrt(totalNoisePower)
signalPowerdBFs = noiseFloor_perBin + object_snr
signalPower = 10**(signalPowerdBFs/10)
signalAmplitude = np.sqrt(signalPower)
signalPhase = np.exp(1j*np.random.uniform(-np.pi, np.pi,size = num_sources))
complex_signal_amplitudes = signalAmplitude*signalPhase

random_freq = np.random.uniform(low=-np.pi, high=np.pi, size = 1)
fft_resol_fact = 2
resol_fact = 0.65
digFreqRes = resol_fact*2*np.pi/num_samples
angResDeg = np.arcsin((digFreqRes/(2*np.pi))*fsSpatial)*180/np.pi
print('Programmed Angular Resolution = {0:.2f} deg'.format(angResDeg))
source_freq = np.array([random_freq, random_freq + digFreqRes])
source_angle_deg = np.arcsin((source_freq/(2*np.pi))*fsSpatial)*180/np.pi
spectrumGridOSRFact = 128 # 32 if we want a less oversampled spectrum evaluation
digital_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/(spectrumGridOSRFact*num_samples))
angleGrid = np.arcsin(((digital_freq_grid/(2*np.pi))*fsSpatial))*180/np.pi
source_signals = np.matmul(np.exp(-1j*np.outer(np.arange(num_samples),source_freq)),complex_signal_amplitudes[:,None])
wgn_noise = np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape)
received_signal = source_signals + wgn_noise
corr_mat_model_order = num_samples//2-2 # must be strictly less than num_samples/2

magnitude_spectrum_fft = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(received_signal,axis=0, n=len(digital_freq_grid))/received_signal.shape[0],axes=0)))
phase_spectrum_fft = np.unwrap(np.angle(np.fft.fftshift(np.fft.fft(received_signal,axis=0, n=len(digital_freq_grid))/received_signal.shape[0],axes=0)))
magnitude_spectrum_fft -= np.amax(magnitude_spectrum_fft,axis=0)[None,:]

magnitude_spectrum_fft_hann = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(received_signal*np.hanning(num_samples)[:,None],axis=0, n=len(digital_freq_grid))/received_signal.shape[0],axes=0)))
magnitude_spectrum_fft_hann -= np.amax(magnitude_spectrum_fft_hann,axis=0)[None,:]

""" SVA"""
osrFact = spectrumGridOSRFact
numFFTOSR = osrFact*num_samples

""" SVA brute force"""
svaSpectralEstimatordB_unoptimal = spatially_variant_apodization_bruteforce(received_signal,numFFTOSR)

""" SVA Optimized"""
svaOptimalComplexSpectrumfftshifted, svaOptimalMagSpectrumdB = spatially_variant_apodization_optimized(received_signal, osrFact)

""" MUSIC"""
pseudo_spectrum = music(received_signal, num_sources, corr_mat_model_order, digital_freq_grid)
pseudo_spectrum = pseudo_spectrum/np.amax(pseudo_spectrum)

""" IAA Recursive """
iterations = 10
spectrum_iaa = iaa_recursive(received_signal, digital_freq_grid, iterations) # recursive IAA
#    spectrum_iaa = spec_est.iaa_recursive_levinson_temp(received_signal, digital_freq_grid, iterations) # under debug
magnitude_spectrum_iaa = np.abs(spectrum_iaa)**2
magnitude_spectrum_iaa = magnitude_spectrum_iaa/np.amax(magnitude_spectrum_iaa)


plt.figure(1,figsize=(20,10))
plt.title('Angle Spectrum.' + ' Native Angular Res (deg) = ' + str(np.round(nativeAngResDeg,2)) + '. Programmed Angular Res (deg) = ' + str(np.round(angResDeg,2)))
plt.plot(angleGrid, magnitude_spectrum_fft, color='blue', label = 'FFT with rectangular window (raised cosine with alpha=0)')
plt.plot(angleGrid, magnitude_spectrum_fft_hann, color='orange',label = 'FFT with hanning window (raised cosine with alpha=0.5)')
plt.plot(angleGrid, svaSpectralEstimatordB_unoptimal, color='violet', linewidth = 6, alpha = 0.5, label = 'Spatially Variant Apodization - brute force')
plt.plot(angleGrid, svaOptimalMagSpectrumdB, color='lime', label = 'Spatially Variant Apodization - optimized')
plt.plot(angleGrid, 10*np.log10(pseudo_spectrum), color='red', label='MUSIC')
plt.plot(angleGrid, 10*np.log10(magnitude_spectrum_iaa), linewidth=2, color='black', label='IAA')
plt.vlines(-source_angle_deg,-80,20, alpha=0.3,label = 'Ground truth')
plt.xlabel('Angle(deg)')
plt.legend()
plt.grid(True)















