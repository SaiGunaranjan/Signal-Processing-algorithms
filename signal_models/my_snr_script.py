# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 11:09:54 2020

@author: Sai Gunaranjan Pelluri
"""


""" This script generates a signal with a user defined SNR post FFT i.e. the snt that is programmed by the user reflects post the FFT
"""
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
noisePowerPerBin_dB = -70;
num_signalSamp = 128;
num_signalSamp_oversamp = 1024;
oversampFact = num_signalSamp_oversamp/num_signalSamp;
tot_noisePower_dB = noisePowerPerBin_dB + 10*np.log10(num_signalSamp);
tot_noisePower = 10**(tot_noisePower_dB/10);
noiseSigma = np.sqrt(tot_noisePower);
snr_dB = 15;#20
sigPow_dB = noisePowerPerBin_dB + snr_dB;
sigWt = 10**(sigPow_dB/20);

sigBin = 26 + np.random.uniform(low=-0.5,high=0.5,size=1)
signal = sigWt*np.exp(1j*2*np.pi*(sigBin/num_signalSamp)*np.arange(num_signalSamp));
noise = noiseSigma/np.sqrt(2)*np.random.randn(num_signalSamp) + 1j*noiseSigma/np.sqrt(2)*np.random.randn(num_signalSamp);
noisy_signal = signal+noise;

noisySignalFFT = np.fft.fft(noisy_signal,n=num_signalSamp_oversamp)/num_signalSamp;
noisySignalSpectrum = 20*np.log10(np.abs(noisySignalFFT));

plt.figure(1,figsize=(20,10));
plt.plot(noisySignalSpectrum,linewidth='3')
plt.grid(True)