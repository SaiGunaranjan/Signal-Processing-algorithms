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
num_samples = 1024
Fs = 40e6
Ts = 1/Fs
chirp_time = num_samples*Ts
chirp_time_samples = np.arange(num_samples)*Ts
Fstart_hz = -Fs/2
Fstop_hz = +Fs/2
slope = (Fstop_hz - Fstart_hz)/chirp_time
freq_grid = np.arange(-num_samples//2, num_samples//2,1)*Fs/num_samples
fre_vs_time = slope*chirp_time_samples + Fstart_hz
chirp_phase = 2*np.pi*(0.5*slope*chirp_time_samples**2 + Fstart_hz*chirp_time_samples + initial_phase_rad)
chirp_signal = 20*np.exp(1j*chirp_phase)
chirp_signal_fft = np.fft.fft(chirp_signal)/num_samples

obj1_freq = 5e6
obj2_freq = 10e6
sinusoid_signal = np.exp(1j*2*np.pi*obj1_freq*chirp_time_samples) + np.exp(1j*2*np.pi*obj2_freq*chirp_time_samples)
sinusoid_signal_fft = np.fft.fft(sinusoid_signal)/num_samples

corrupted_signal = sinusoid_signal + chirp_signal
corrupted_signal_fft = np.fft.fft(corrupted_signal)/num_samples

corrupted_signal_firsthalf = corrupted_signal[0:num_samples//2]
corrupted_signal_secondhalf = corrupted_signal[num_samples//2::]

corrupted_signal_firsthalf_fft = np.fft.fft(corrupted_signal_firsthalf,n=1024)/num_samples
corrupted_signal_secondhalf_fft = np.fft.fft(corrupted_signal_secondhalf,n=1024)/num_samples

plt.figure(1)
plt.title('chirp freq vs time')
plt.plot(chirp_time_samples/(1e-6),fre_vs_time/1e6)
plt.xlabel('Time (us)')
plt.ylabel('Freq (GHz)')
plt.grid(True)

plt.figure(2)
plt.subplot(131)
plt.title('FFT of chirp signal')
plt.plot(freq_grid/1e6, 20*np.log10(np.fft.fftshift(np.abs(chirp_signal_fft))))
plt.xlabel('Freq(MHz)')
plt.grid(True)
plt.subplot(132)
plt.title('FFT of sinusoid signal')
plt.plot(freq_grid/1e6, 20*np.log10(np.fft.fftshift(np.abs(sinusoid_signal_fft))))
plt.xlabel('Freq(MHz)')
plt.grid(True)
plt.subplot(133)
plt.title('FFT of interference corrupted signal')
plt.plot(freq_grid/1e6, 20*np.log10(np.fft.fftshift(np.abs(corrupted_signal_fft))))
plt.xlabel('Freq(MHz)')
plt.grid(True)


plt.figure(3)
plt.subplot(121)
plt.title('FFT using first half samples')
plt.plot(freq_grid/1e6, 20*np.log10(np.fft.fftshift(np.abs(corrupted_signal_firsthalf_fft))))
plt.xlabel('Freq(MHz)')
plt.grid(True)

plt.subplot(122)
plt.title('FFT using second half samples')
plt.plot(freq_grid/1e6, 20*np.log10(np.fft.fftshift(np.abs(corrupted_signal_secondhalf_fft))))
plt.xlabel('Freq(MHz)')
plt.grid(True)

