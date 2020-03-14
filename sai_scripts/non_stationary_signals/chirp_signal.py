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
chirp_time = 80e-6 # us
slope = 25e6/1e-6 # MHz/us
Fs = 200e9 # GHz
Ts = 1/Fs
Fstart_hz = 77e9 # GHz
num_samples = np.int32(chirp_time//Ts)
time_s = np.arange(num_samples)*Ts
freq_grid = np.arange(-num_samples//2, num_samples//2,1)*Fs/num_samples
fre_vs_time = slope*time_s + Fstart_hz
#chirp_phase = 2*np.pi*(0.5*slope*time_s**2 + Fstart_hz*time_s) + initial_phase_rad;
chirp_phase = 2*np.pi*np.cumsum(fre_vs_time)*Ts + initial_phase_rad
chirp_signal = np.exp(1j*chirp_phase)
chirp_signal_InstFreq = (np.diff(np.unwrap(np.angle(chirp_signal)))/(2*np.pi))/Ts;
chirp_signal_fft = np.fft.fft(chirp_signal)/num_samples

ang_var_deg = 10;
ang_var_rad = ang_var_deg*np.pi/180;
phase_noise = np.exp(1j*(np.random.uniform(low=-ang_var_rad,high=ang_var_rad,size=num_samples)));
chirp_signal_PhaseNoise = np.exp(1j*chirp_phase)*phase_noise;
chirp_signal_PhaseNoise_InstFreq = (np.diff(np.unwrap(np.angle(chirp_signal_PhaseNoise)))/(2*np.pi))/Ts;
chirp_signal_PhaseNoise_fft = np.fft.fft(chirp_signal_PhaseNoise)/num_samples

plt.figure(1,figsize=(20,9))
plt.title('Target chirp signal: Freq vs Time')
plt.plot(time_s/(1e-6),fre_vs_time/1e9)
plt.xlabel('Time (us)')
plt.ylabel('Freq (GHz)')
plt.grid(True)


plt.figure(2,figsize=(20,9))
plt.subplot(231)
plt.title('Instantaneous Freq of clean lin. chirp signal')
plt.plot(time_s[0:-1]/(1e-6),chirp_signal_InstFreq/1e9)
plt.xlabel('Time (us)')
plt.ylabel('Freq (GHz)')
plt.grid(True)
plt.subplot(232)
plt.title('Clean Chirp signal Magnitude spectrum (dB)')
plt.plot(freq_grid/1e9, 20*np.log10(np.fft.fftshift(np.abs(chirp_signal_fft))))
plt.xlabel('Freq(GHz)')
plt.grid(True)
plt.subplot(233)
plt.title('Clean Chirp signal Phase spectrum (deg)')
plt.plot(freq_grid/1e9, (180/np.pi)*(np.unwrap(np.angle(np.fft.fftshift(chirp_signal_fft)))))
plt.xlabel('Freq(GHz)')
plt.grid(True)

plt.subplot(234)
plt.title('Instantaneous Freq of noisy lin. chirp signal')
plt.plot(time_s[0:-1]/(1e-6),chirp_signal_PhaseNoise_InstFreq/1e9)
plt.xlabel('Time (us)')
plt.ylabel('Freq (GHz)')
plt.grid(True)
plt.subplot(235)
plt.title('phase Noise corr. Chirp signal Magnitude spectrum (dB)')
plt.plot(freq_grid/1e9, 20*np.log10(np.fft.fftshift(np.abs(chirp_signal_PhaseNoise_fft))))
plt.xlabel('Freq(GHz)')
plt.grid(True)
plt.subplot(236)
plt.title('phase Noise corr Chirp signal Phase spectrum (deg)')
plt.plot(freq_grid/1e9, (180/np.pi)*(np.unwrap(np.angle(np.fft.fftshift(chirp_signal_PhaseNoise_fft)))))
plt.xlabel('Freq(GHz)')
plt.grid(True)
