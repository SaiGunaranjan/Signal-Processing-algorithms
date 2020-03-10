# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 12:21:07 2019

@author: Sai Gunaranjan Pelluri
"""

import numpy as np;
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


plt.close('all');
Amp = 1/10;
Ts = 36.1e-6
Fs = 1/Ts;
fo= 1024;
dig_freq = (fo/Fs)*2*np.pi;
numsamp = 130*(2**0);
window = 1#np.hamming(numsamp)
#sig = np.arange(numsamp) + window*np.sin(dig_freq*np.arange(numsamp));
sig = (2*np.pi/numsamp)*np.arange(numsamp) + np.sin(dig_freq*np.arange(numsamp));
N = numsamp;
sig_fft = np.fft.fft(sig,n=N)/len(sig);
fm_sig = np.exp(1j*sig) + 0*1;
fm_sig = fm_sig*np.hamming(N);
fm_sig_fft = np.fft.fft(fm_sig,n=N)/len(fm_sig);
fft_axis = np.arange(-Fs/2,Fs/2,Fs/N);
plt.figure(1);
plt.subplot(2,2,1);
plt.title('Phase signal')
plt.plot(sig);
plt.xlabel('Num samples');
plt.ylabel('Amp');
plt.grid(True);
plt.subplot(2,2,2);
plt.title('Real part of the FM signal')
plt.plot(Ts*np.arange(numsamp),np.real(fm_sig));
plt.xlabel('Time (s)');
plt.ylabel('Amp');
plt.grid(True);


plt.subplot(2,2,3);
plt.title('FFT of the phase')
plt.plot(fft_axis,20*np.log10(np.abs(np.fft.fftshift(sig_fft))),'-o');
plt.xlabel('Freq (Hz)');
plt.ylabel('Power (dB)');
plt.grid(True);
plt.subplot(2,2,4);
plt.title('FFT of phasor')
plt.plot(fft_axis,20*np.log10(np.abs(np.fft.fftshift(fm_sig_fft))),'-o');
plt.xlabel('Freq (Hz)');
plt.ylabel('Power (dB)');
plt.grid(True);

if 0:
    window_size = 32;
    freq_axis, time_axis, stft = spectrogram(fm_sig, fs=Fs,  \
                             nperseg=window_size, noverlap=window_size//2, nfft=window_size*2);
    plt.figure(2);
    plt.pcolormesh(time_axis, freq_axis, stft)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

