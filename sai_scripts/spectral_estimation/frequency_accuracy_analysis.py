# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 21:06:12 2019

@author: Sai Gunaranjan Pelluri
"""
import numpy as np
import spectral_estimation_lib as spec_est
import matplotlib.pyplot as plt
from time import time


""" Seems to have a bug. Need to debug this as well"""
plt.close('all')

SNR = np.arange(-5,25,2)
num_samples = 256#1024
signal_power = 1
num_montecarlo = 10#100
num_sources = 1
corr_mat_model_order = 100 # corr_mat_model_order : must be strictly less than half the signal length
digital_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/(100*num_samples))
rmse_fft_snr = []
rmse_music_snr = []
rmse_esprit_snr = []
rmse_capon_snr = []
rmse_apes_snr = []
tstart = time()
for snr in SNR:
    noise_power = 10**(-snr/10)
    noise_sigma = np.sqrt(noise_power)
    error_fft = []
    error_music = []
    error_esprit = []
    error_capon = []
    error_apes = []
    for iteration in np.arange(num_montecarlo):
        dig_freq = np.random.uniform(-np.pi, np.pi)
        signal = np.exp(1j*dig_freq*np.arange(num_samples))/np.sqrt(num_samples)
        noise = np.random.normal(0,noise_sigma/np.sqrt(2),num_samples) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),num_samples)
        noisy_signal = (signal + noise)[:,None]
        fft_index = np.argmax(np.abs(np.fft.fft(noisy_signal,axis=0)))
        est_dig_freq_fft = (2*np.pi/num_samples)*(fft_index if fft_index <= num_samples//2 else fft_index-num_samples)
        pseudo_spectrum = spec_est.music_backward(noisy_signal, num_sources, corr_mat_model_order, digital_freq_grid)
        est_dig_freq_music = float(digital_freq_grid[np.argsort(pseudo_spectrum)[-num_sources::]])
        est_dig_freq_esprit = float(-1*spec_est.esprit_backward(noisy_signal, num_sources, corr_mat_model_order))
        psd_b = spec_est.capon_backward(noisy_signal, corr_mat_model_order, digital_freq_grid)
        est_dig_freq_capon = digital_freq_grid[np.argmax(psd_b)]
        magnitude_spectrum = spec_est.apes(noisy_signal, corr_mat_model_order, digital_freq_grid)
        est_dig_freq_apes = digital_freq_grid[np.argmax(magnitude_spectrum)]
        error_fft.append(dig_freq-est_dig_freq_fft)
        error_music.append(dig_freq-est_dig_freq_music)
        error_esprit.append(dig_freq-est_dig_freq_esprit)
        error_capon.append(dig_freq-est_dig_freq_capon)
        error_apes.append(dig_freq-est_dig_freq_apes)
    rmse_fft =  np.sqrt(np.sum(np.array(error_fft)**2)/num_montecarlo)
    rmse_music =  np.sqrt(np.sum(np.array(error_music)**2)/num_montecarlo)
    rmse_esprit =  np.sqrt(np.sum(np.array(error_esprit)**2)/num_montecarlo)
    rmse_capon =  np.sqrt(np.sum(np.array(error_capon)**2)/num_montecarlo)
    rmse_apes =  np.sqrt(np.sum(np.array(error_apes)**2)/num_montecarlo)
    rmse_fft_snr.append(rmse_fft)
    rmse_music_snr.append(rmse_music)
    rmse_esprit_snr.append(rmse_esprit)
    rmse_capon_snr.append(rmse_capon)
    rmse_apes_snr.append(rmse_apes)
tend = time()
print("Time elapsed = {0:.1f} secs".format(tend-tstart))        
plt.figure(1)
plt.plot(SNR, np.array(rmse_fft_snr),'o-',label='FFT')
plt.plot(SNR, np.array(rmse_music_snr),'o-',label='MUSIC')
plt.plot(SNR, np.array(rmse_esprit_snr),'o-',label='ESPRIT')
plt.plot(SNR, np.array(rmse_capon_snr),'o-',label='CAPON')
plt.plot(SNR, np.array(rmse_apes_snr),'o-',label='APES')
plt.xlabel('SNR (dB)')
plt.ylabel('RMSE in rad/sample')
plt.legend()
plt.grid(True)
