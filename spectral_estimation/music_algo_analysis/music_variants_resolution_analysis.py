# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 13:08:31 2023

@author: Sai Gunaranjan
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..//')

from spectral_estimation_lib import music_backward as music
from spectral_estimation_lib import music_snapshots#, music_denso

np.random.seed(5)

plt.close('all')
num_samples = 6#8#32
c = 3e8
fc = 79e9
lamda = c/fc
mimoSpacing = lamda/2
fsSpatial = lamda/mimoSpacing
nativeAngResDeg = np.arcsin(fsSpatial/num_samples)*180/np.pi
print('Native Angular Resolution = {0:.2f} deg'.format(nativeAngResDeg))
num_sources = 2
numSnapshots = 50
object_snr = np.array([40,35])
noiseFloordB = -100
noise_power_db = noiseFloordB + 10*np.log10(num_samples)
noise_variance = 10**(noise_power_db/10)
noise_sigma = np.sqrt(noise_variance)
weights = np.sqrt(10**((noiseFloordB + object_snr)/10))
signal_phases = np.exp(1j*np.random.uniform(low=-np.pi, high = np.pi, size = numSnapshots*num_sources))
signal_phases = signal_phases.reshape(num_sources,numSnapshots)
complex_signal_amplitudes = weights[:,None]*signal_phases
random_freq = np.random.uniform(low=-np.pi, high=np.pi, size = 1)
fft_resol_fact = 2
resol_fact = 0.1#0.5#0.65 # 0.65
digFreqRes = resol_fact*((2*np.pi)/num_samples)
angResDeg = np.arcsin((digFreqRes/(2*np.pi))*fsSpatial)*180/np.pi
print('Programmed Angular Resolution = {0:.2f} deg'.format(angResDeg))
source_freq = np.array([0, 0 + digFreqRes])
source_angle_deg = np.arcsin((source_freq/(2*np.pi))*fsSpatial)*180/np.pi
spectrumGridOSRFact = 32#256#128 # 32 if we want a less oversampled spectrum evaluation
digital_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/(spectrumGridOSRFact*num_samples))
angleGrid = np.arcsin(((digital_freq_grid/(2*np.pi))*fsSpatial))*180/np.pi
signalPhase = np.exp(1j*np.outer(np.arange(num_samples),source_freq))
source_signals = signalPhase @ complex_signal_amplitudes

wgn_noise = (noise_sigma/np.sqrt(2))*np.random.randn(source_signals.shape[0] * numSnapshots) + 1j*(noise_sigma/np.sqrt(2))*np.random.randn(source_signals.shape[0] * numSnapshots)
received_signal = source_signals + wgn_noise.reshape(-1,numSnapshots)
corr_mat_model_order = num_samples//2-1 # must be strictly less than num_samples/2. Should ideally be num_samples/2 - 2

magnitude_spectrum_fft = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(received_signal*np.hanning(num_samples)[:,None],axis=0, n=len(digital_freq_grid))/received_signal.shape[0],axes=0)))
magnitude_spectrum_fft -= np.amax(magnitude_spectrum_fft,axis=0)[None,:]

""" MUSIC snaphsots"""
pseudo_spectrum = music_snapshots(received_signal, num_sources, num_samples, digital_freq_grid)
pseudo_spectrum = pseudo_spectrum/np.amax(pseudo_spectrum)

""" MUSIC sliding window"""
pseudo_spectrum_avg = music(received_signal[:,0][:,None], num_sources, corr_mat_model_order, digital_freq_grid)
pseudo_spectrum_avg = pseudo_spectrum_avg/np.amax(pseudo_spectrum_avg)

# """ MUSIC denso - spatial smoothing"""
# pseudo_spectrum_denso = music_denso(received_signal, num_sources, num_samples, digital_freq_grid)
# pseudo_spectrum_denso = pseudo_spectrum_denso/np.amax(pseudo_spectrum_denso)



plt.figure(1,figsize=(20,10),dpi=200)
plt.title('Num MIMO samples = ' + str(num_samples) + '. Native Angular Res (deg) = ' + str(np.round(nativeAngResDeg,2)) + '. Programmed Angular Res (deg) = ' + str(np.round(angResDeg,2)))
plt.plot(angleGrid, magnitude_spectrum_fft[:,0], label = 'FFT')
# plt.plot(angleGrid, 20*np.log10(pseudo_spectrum_avg), label='MUSIC sliding window')
plt.plot(angleGrid, 20*np.log10(pseudo_spectrum), label='MUSIC snapshots = {}'.format(numSnapshots))
# plt.plot(angleGrid, 20*np.log10(pseudo_spectrum_denso), label='MUSIC denso')
plt.vlines(source_angle_deg,-100,5, alpha=1,color='black',ls='dashed',label = 'Ground truth')
plt.xlabel('Angle(deg)')
plt.legend()
plt.grid(True)







