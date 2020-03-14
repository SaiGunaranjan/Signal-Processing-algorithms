# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 12:02:41 2019

@author: Sai Gunaranjan Pelluri
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
initial_phase_deg = 30
initial_phase_rad = initial_phase_deg*np.pi/180
chirp_time = 36e-6 # us
slope = 25e6/(1e-6) # MHz/us
Fs = 2*200e9
Ts = 1/Fs
Fstart_hz = 77e9 # GHz
num_samples = np.int32(chirp_time//Ts)
time_s = np.arange(num_samples)*Ts

downSamplingfact = np.int32(1e4)

distObj = 20;
lightSpeed = 3e8
objectTimeDelay = 2*distObj/lightSpeed;
objectSamplesDelay = np.int32(np.floor(objectTimeDelay//Ts))
trueBasebandFreq = slope*objectTimeDelay;

freq_grid = np.arange(-num_samples//2, num_samples//2,1)*Fs/num_samples
fre_vs_time = slope*time_s + Fstart_hz

chirp_phase = 2*np.pi*(0.5*slope*time_s**2 + Fstart_hz*time_s) + initial_phase_rad;
#chirp_phase = 2*np.pi*np.cumsum(fre_vs_time)*Ts + initial_phase_rad

ang_var_deg = 10;
ang_var_rad = ang_var_deg*np.pi/180;
phase_noise = np.exp(1j*(np.random.uniform(low=-ang_var_rad,high=ang_var_rad,size=num_samples)));
clean_chirp_signal = np.exp(1j*chirp_phase);
chirp_signal = clean_chirp_signal*phase_noise;

delayedCleanChirpSignal = np.hstack((np.zeros((objectSamplesDelay)), clean_chirp_signal))[0:num_samples];
delayedChirpSignal = np.hstack((np.zeros((objectSamplesDelay)), chirp_signal))[0:num_samples];

downConv_CleanIFSignal = clean_chirp_signal*np.conj(delayedCleanChirpSignal);
downConv_CleanIFSignal = downConv_CleanIFSignal[objectSamplesDelay::];
downSamp_downConv_CleanIFSignal = downConv_CleanIFSignal[0::downSamplingfact];

downConv_IFSignal = chirp_signal*np.conj(delayedChirpSignal);
downConv_IFSignal = downConv_IFSignal[objectSamplesDelay::];
downSamp_downConv_IFSignal = downConv_IFSignal[0::downSamplingfact];

num_samp_IFSignal = len(downSamp_downConv_IFSignal)
window_fn = np.hamming(num_samp_IFSignal)  #np.ones(num_samp_IFSignal) #

downSamp_downConv_CleanIFSignal = downSamp_downConv_CleanIFSignal*window_fn;
downSamp_downConv_IFSignal = downSamp_downConv_IFSignal*window_fn;


nfft = np.int32(2**(np.ceil(np.log2(num_samp_IFSignal))));

downSamp_downConv_CleanIFSignal_fft = np.fft.fft(downSamp_downConv_CleanIFSignal,n=nfft)/num_samp_IFSignal;
downSamp_downConv_IFSignal_fft = np.fft.fft(downSamp_downConv_IFSignal,n=nfft)/num_samp_IFSignal;

baseBandFs = Fs/downSamplingfact;
freq_grid_IFSignal = np.arange(-nfft//2, nfft//2,1)*baseBandFs/nfft

plt.figure(1,figsize=(20,10))
plt.subplot(121)
plt.title('Clean IF signal Magnitude spectrum (dB)')
plt.plot(freq_grid_IFSignal/1e6, 20*np.log10(np.fft.fftshift(np.abs(downSamp_downConv_CleanIFSignal_fft))))
plt.axvline(trueBasebandFreq/1e6,color='k',alpha=0.8, linestyle='dashed', linewidth=2)
plt.xlabel('Freq(MHz)')
plt.grid(True)
plt.subplot(122)
plt.title('+/- ' + str(ang_var_deg) + ' deg' +' phase Noise corr. IF signal Magnitude spectrum (dB)')
plt.plot(freq_grid_IFSignal/1e6, 20*np.log10(np.fft.fftshift(np.abs(downSamp_downConv_IFSignal_fft))))
plt.axvline(trueBasebandFreq/1e6,color='k',alpha=0.8, linestyle='dashed', linewidth=2,label='True Frequency');
plt.legend();
plt.xlabel('Freq(MHz)')
plt.grid(True)









