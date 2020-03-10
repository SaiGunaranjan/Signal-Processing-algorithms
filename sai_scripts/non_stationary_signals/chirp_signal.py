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
chirp_time = 80e-6
slope = 25e12
Fs = 200e9
Ts = 1/Fs
Fstart_hz = -77e9
num_samples = chirp_time//Ts
time_s = np.arange(num_samples)*Ts
freq_grid = np.arange(-num_samples//2, num_samples//2,1)*Fs/num_samples
fre_vs_time = slope*time_s + Fstart_hz

chirp_phase = 2*np.pi*(0.5*slope*time_s**2 + Fstart_hz*time_s + initial_phase_rad)
chirp_signal = np.exp(1j*chirp_phase)
chirp_signal_fft = np.fft.fft(chirp_signal)
plt.figure(1)
plt.plot(time_s/(1e-6),fre_vs_time/1e9)
plt.xlabel('Time (us)')
plt.ylabel('Freq (GHz)')
plt.figure(2)
plt.plot(freq_grid/1e9, 20*np.log10(np.fft.fftshift(np.abs(chirp_signal_fft))))
plt.xlabel('Freq(GHz)')
plt.figure(3)
plt.plot(freq_grid/1e9, 180/np.pi*(np.unwrap(np.angle(np.fft.fftshift(chirp_signal_fft)))))
plt.xlabel('Freq(GHz)')
