# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 18:19:28 2020

@author: Steradian Semi
"""

""" This script generates the plots for how the spectrum of s signal looks like 
    in the following conditions:
    1. signal on/off bin.
    2. With and without zero-padding
    3. With and without phase noise.
    4. With and without windowing
    """

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

num_samp = 512;
num_fft = 2*num_samp;
freq_grid = np.arange(-num_samp//2, num_samp//2);
freq_grid_interp = np.arange(-num_fft//2, num_fft//2);
freq_bin = 30 + 0.3; # Check with integer and non-integer
signal = np.exp(1j*2*np.pi*(freq_bin/num_samp)*np.arange(num_samp));
ang_var_deg = 10;
ang_var_rad = ang_var_deg*np.pi/180;
phase_noise = np.exp(1j*(np.random.uniform(low=-ang_var_rad,high=ang_var_rad,size=num_samp)));
window = np.hamming(num_samp);
signal_PhaseNoise = signal*phase_noise;
signal_window = signal*window;
signal_PhaseNoiseWindow = signal*phase_noise*window;

signal_fft = np.fft.fftshift(np.fft.fft(signal,n=num_samp))/num_samp;
signal_window_fft = np.fft.fftshift(np.fft.fft(signal_window,n=num_samp))/num_samp;
signal_PhaseNoise_fft = np.fft.fftshift(np.fft.fft(signal_PhaseNoise,n=num_samp))/num_samp;
signal_PhaseNoiseWindow_fft = np.fft.fftshift(np.fft.fft(signal_PhaseNoiseWindow,n=num_samp))/num_samp;

signal_fft_ext = np.fft.fftshift(np.fft.fft(signal,n=num_fft))/num_samp;
signal_window_fft_ext = np.fft.fftshift(np.fft.fft(signal_window,n=num_fft))/num_samp;
signal_PhaseNoise_fft_ext = np.fft.fftshift(np.fft.fft(signal_PhaseNoise,n=num_fft))/num_samp;
signal_PhaseNoiseWindow_fft_ext = np.fft.fftshift(np.fft.fft(signal_PhaseNoiseWindow,n=num_fft))/num_samp;

plt.figure(1,figsize=(30,12));

plt.subplot(241);
plt.title('Clean signal spectrum');
plt.plot(freq_grid,20*np.log10(np.abs(signal_fft)));
plt.grid(True);
plt.xlabel('Freq bin index');

plt.subplot(242);
plt.title('Windowed Clean  signal spectrum');
plt.plot(freq_grid,20*np.log10(np.abs(signal_window_fft)));
plt.grid(True);
plt.xlabel('Freq bin index');

plt.subplot(243);
plt.title('Phase noise signal spectrum');
plt.plot(freq_grid,20*np.log10(np.abs(signal_PhaseNoise_fft)));
plt.grid(True);
plt.xlabel('Freq bin index');

plt.subplot(244);
plt.title('Windowed Phase noise signal spectrum');
plt.plot(freq_grid,20*np.log10(np.abs(signal_PhaseNoiseWindow_fft)));
plt.grid(True);
plt.xlabel('Freq bin index');
#plt.ylim(0.-350)

plt.subplot(245);
plt.title('Extended Clean signal spectrum');
plt.plot(freq_grid_interp,20*np.log10(np.abs(signal_fft_ext)));
plt.grid(True);
plt.xlabel('Freq bin index');

plt.subplot(246);
plt.title('Extended Windowed Clean signal spectrum');
plt.plot(freq_grid_interp,20*np.log10(np.abs(signal_window_fft_ext)));
plt.grid(True);
plt.xlabel('Freq bin index');

plt.subplot(247);
plt.title('Extended Phase noise signal spectrum');
plt.plot(freq_grid_interp,20*np.log10(np.abs(signal_PhaseNoise_fft_ext)));
plt.grid(True);
plt.xlabel('Freq bin index');

plt.subplot(248);
plt.title('Extended Windowed Phase noise signal spectrum');
plt.plot(freq_grid_interp,20*np.log10(np.abs(signal_PhaseNoiseWindow_fft_ext)));
plt.grid(True);
plt.xlabel('Freq bin index');