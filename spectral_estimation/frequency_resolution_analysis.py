# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 21:06:12 2019

@author: Sai Gunaranjan Pelluri
"""
import numpy as np
import spectral_estimation_lib as spec_est
import matplotlib.pyplot as plt
from time import time


"""There seems to be a bug in this code. The plots don't seem to be correct """

plt.close('all')

SNR = np.arange(25,26)#np.arange(-5,25,2)
num_samples = 256#1024
signal_power = 1
num_montecarlo = 3#100
num_sources = 2
corr_mat_model_order = 100 # corr_mat_model_order : must be strictly less than half the signal length
fft_freq_resol = 2*np.pi/num_samples
min_dig_freq_step_size = fft_freq_resol/10
digital_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/(100*num_samples))
dig_freq_steps = np.arange(min_dig_freq_step_size,fft_freq_resol,min_dig_freq_step_size)

rmse_fft_snr = np.zeros((0,len(dig_freq_steps)))
rmse_music_snr = np.zeros((0,len(dig_freq_steps)))
rmse_esprit_snr = np.zeros((0,len(dig_freq_steps)))
rmse_capon_snr = np.zeros((0,len(dig_freq_steps)))
rmse_apes_snr = np.zeros((0,len(dig_freq_steps)))

rmse_fft_snr_sec_source = np.zeros((0,len(dig_freq_steps)))
rmse_music_snr_sec_source = np.zeros((0,len(dig_freq_steps)))
rmse_esprit_snr_sec_source = np.zeros((0,len(dig_freq_steps)))
rmse_capon_snr_sec_source = np.zeros((0,len(dig_freq_steps)))
rmse_apes_snr_sec_source = np.zeros((0,len(dig_freq_steps)))
tstart = time()
for snr in SNR:
    noise_power = 10**(-snr/10)
    noise_sigma = np.sqrt(noise_power)
    error_fft = []
    error_music = []
    error_esprit = []
    error_capon = []
    error_apes = []
    error_fft_sec_source = []
    error_music_sec_source = []
    error_esprit_sec_source = []
    error_capon_sec_source = []
    error_apes_sec_source = []
    
    rmse_fft_res = []
    rmse_music_res = []
    rmse_esprit_res = []
    rmse_capon_res = []
    rmse_apes_res = []
    rmse_fft_res_sec_source = []
    rmse_music_res_sec_source = []
    rmse_esprit_res_sec_source = []
    rmse_capon_res_sec_source = []
    rmse_apes_res_sec_source = []
    
    for dig_freq_delta in dig_freq_steps:
        dig_freq = np.random.uniform(-np.pi, np.pi)
        second_source_dig_freq = dig_freq + dig_freq_delta
        signal1 = np.exp(1j*dig_freq*np.arange(num_samples))
        signal2 = np.exp(1j*second_source_dig_freq*np.arange(num_samples))
        signal = (signal1 + signal2)/np.sqrt(num_samples)
        for iteration in np.arange(num_montecarlo):
            noise = np.random.normal(0,noise_sigma/np.sqrt(2),num_samples) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),num_samples)
            noisy_signal = (signal + noise)[:,None]
            fft_indices = np.argsort(np.abs(np.fft.fft(noisy_signal,axis=0)),axis=0)[-num_sources::]
            est_dig_freq_fft = (2*np.pi/num_samples)*np.array([fft_index if fft_index <= num_samples//2 else fft_index-num_samples for fft_index in fft_indices])
            pseudo_spectrum = spec_est.music_backward(noisy_signal, num_sources, corr_mat_model_order, digital_freq_grid)
            est_dig_freq_music = digital_freq_grid[np.argsort(pseudo_spectrum)[-num_sources::]]
            est_dig_freq_esprit = -1*spec_est.esprit_backward(noisy_signal, num_sources, corr_mat_model_order)
            psd_b = spec_est.capon_backward(noisy_signal, corr_mat_model_order, digital_freq_grid)
            est_dig_freq_capon = digital_freq_grid[np.argsort(psd_b)[-num_sources::]]
            spectrum = spec_est.apes(noisy_signal, corr_mat_model_order, digital_freq_grid)
            magnitude_spectrum = np.abs(spectrum)
            est_dig_freq_apes = digital_freq_grid[np.argsort(magnitude_spectrum)[-num_sources::]]
            
            error_fft.append(dig_freq-np.amin(est_dig_freq_fft))
            error_music.append(dig_freq-np.amin(est_dig_freq_music))
            error_esprit.append(dig_freq-np.amin(est_dig_freq_esprit))
            error_capon.append(dig_freq-np.amin(est_dig_freq_capon))
            error_apes.append(dig_freq-np.amin(est_dig_freq_apes))
    
            error_fft_sec_source.append(second_source_dig_freq-np.amax(est_dig_freq_fft))
            error_music_sec_source.append(second_source_dig_freq-np.amax(est_dig_freq_music))
            error_esprit_sec_source.append(second_source_dig_freq-np.amax(est_dig_freq_esprit))
            error_capon_sec_source.append(second_source_dig_freq-np.amax(est_dig_freq_capon))
            error_apes_sec_source.append(second_source_dig_freq-np.amax(est_dig_freq_apes))
        
        rmse_fft =  np.sqrt(np.sum(np.array(error_fft)**2)/num_montecarlo)
        rmse_music =  np.sqrt(np.sum(np.array(error_music)**2)/num_montecarlo)
        rmse_esprit =  np.sqrt(np.sum(np.array(error_esprit)**2)/num_montecarlo)
        rmse_capon =  np.sqrt(np.sum(np.array(error_capon)**2)/num_montecarlo)
        rmse_apes =  np.sqrt(np.sum(np.array(error_apes)**2)/num_montecarlo)
    
        rmse_fft_sec_source =  np.sqrt(np.sum(np.array(error_fft_sec_source)**2)/num_montecarlo)
        rmse_music_sec_source =  np.sqrt(np.sum(np.array(error_music_sec_source)**2)/num_montecarlo)
        rmse_esprit_sec_source =  np.sqrt(np.sum(np.array(error_esprit_sec_source)**2)/num_montecarlo)
        rmse_capon_sec_source =  np.sqrt(np.sum(np.array(error_capon_sec_source)**2)/num_montecarlo)
        rmse_apes_sec_source =  np.sqrt(np.sum(np.array(error_apes_sec_source)**2)/num_montecarlo)
        
        rmse_fft_res.append(rmse_fft)
        rmse_music_res.append(rmse_music)
        rmse_esprit_res.append(rmse_esprit)
        rmse_capon_res.append(rmse_capon)
        rmse_apes_res.append(rmse_apes)
    
    
        rmse_fft_res_sec_source.append(rmse_fft_sec_source)
        rmse_music_res_sec_source.append(rmse_music_sec_source)
        rmse_esprit_res_sec_source.append(rmse_esprit_sec_source)
        rmse_capon_res_sec_source.append(rmse_capon_sec_source)
        rmse_apes_res_sec_source.append(rmse_apes_sec_source)
        
        
    
    rmse_fft_snr = np.vstack((rmse_fft_snr,rmse_fft_res))
    rmse_music_snr = np.vstack((rmse_music_snr,rmse_music_res))
    rmse_esprit_snr = np.vstack((rmse_esprit_snr,rmse_esprit_res))
    rmse_capon_snr = np.vstack((rmse_capon_snr,rmse_capon_res))
    rmse_apes_snr = np.vstack((rmse_apes_snr,rmse_apes_res))


    rmse_fft_snr_sec_source = np.vstack((rmse_fft_snr_sec_source,rmse_fft_res_sec_source))
    rmse_music_snr_sec_source = np.vstack((rmse_music_snr_sec_source,rmse_music_res_sec_source))
    rmse_esprit_snr_sec_source = np.vstack((rmse_esprit_snr_sec_source,rmse_esprit_res_sec_source))
    rmse_capon_snr_sec_source = np.vstack((rmse_capon_snr_sec_source,rmse_capon_res_sec_source))
    rmse_apes_snr_sec_source = np.vstack((rmse_apes_snr_sec_source,rmse_apes_res_sec_source))
    
tend = time()
print("Time elapsed = {0:.1f} secs".format(tend-tstart))


      
plt.figure(1)
plt.subplot(121)
plt.title('First source')
#plt.plot(dig_freq_steps, rmse_fft_snr.squeeze(),'o-',label='FFT')
#plt.plot(dig_freq_steps, rmse_music_snr.squeeze(),'o-',label='MUSIC')
#plt.plot(dig_freq_steps, rmse_esprit_snr.squeeze(),'o-',label='ESPRIT')
plt.plot(dig_freq_steps, rmse_capon_snr.squeeze(),'o-',label='CAPON')
plt.plot(dig_freq_steps, rmse_apes_snr.squeeze(),'o-',label='APES')
plt.xlabel('Digital Freq Resol (rad/samp)')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)

plt.subplot(122)
plt.title('Second source')
#plt.plot(dig_freq_steps, rmse_fft_snr_sec_source.squeeze(),'o-',label='FFT')
#plt.plot(dig_freq_steps, rmse_music_snr_sec_source.squeeze(),'o-',label='MUSIC')
#plt.plot(dig_freq_steps, rmse_esprit_snr_sec_source.squeeze(),'o-',label='ESPRIT')
plt.plot(dig_freq_steps, rmse_capon_snr_sec_source.squeeze(),'o-',label='CAPON')
plt.plot(dig_freq_steps, rmse_apes_snr_sec_source.squeeze(),'o-',label='APES')
plt.xlabel('Digital Freq Resol (rad/samp)')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
