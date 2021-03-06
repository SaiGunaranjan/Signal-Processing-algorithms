# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 20:51:46 2019

@author: Sai Gunaranjan Pelluri

This is a script to show that coherant averaging gives better SNR improvement over non-coherant averaging. 
In both cases, the signal power is the same but the noise power increases in non-coherant averaging.

"""

import numpy as np
import matplotlib.pyplot as plt


plt.close('all')
num_samp_range = 1024*2*2*2*2
fre_ind_range = np.random.randint(num_samp_range)
omega_range = (2*np.pi/num_samp_range)*fre_ind_range
num_samp_dopp = 256
fre_ind_dopp = np.random.randint(num_samp_dopp)
omega_dopp = 0#(2*np.pi/num_samp_dopp)*fre_ind_dopp
range_signal = np.exp(1j*omega_range*np.arange(num_samp_range))[:,None]
range_signal_power = (np.linalg.norm(range_signal,2)**2/num_samp_range)
SNR = 20
noise_power = range_signal_power*10**(-SNR/10)
noiseFloorPerBin = noise_power/num_samp_range
noise_sigma = np.sqrt(noise_power)
#noise_signal = (1/np.sqrt(num_samp_dopp))*((noise_sigma/np.sqrt(2))*np.random.randn(num_samp_range,num_samp_dopp) + 1j*(noise_sigma/np.sqrt(2))*np.random.randn(num_samp_range,num_samp_dopp))
noise_signal = ((noise_sigma/np.sqrt(2))*np.random.randn(num_samp_range,num_samp_dopp) + 1j*(noise_sigma/np.sqrt(2))*np.random.randn(num_samp_range,num_samp_dopp))
range_signal_noise = range_signal + noise_signal
dopp_signal = np.exp(1j*omega_dopp*np.arange(num_samp_dopp))
signal = range_signal_noise*dopp_signal
signal_range_fft = np.fft.fft(signal,axis=0)/num_samp_range
non_coh_avg = np.sqrt(np.sum(np.abs(signal_range_fft)**2,axis=1)/num_samp_dopp)
coh_avg = np.fft.fft(signal_range_fft,axis=1)[:,0]/num_samp_dopp
est_noise_power_range_fft = 10*np.log10((np.sum(np.abs(signal_range_fft[:,0])**2) - np.amax(np.abs(signal_range_fft[:,0]))**2))
est_noise_power_non_coh_avg = 10*np.log10((np.sum(non_coh_avg**2) - np.amax(non_coh_avg)**2))
est_noise_power_coh_avg = 10*np.log10((np.sum(np.abs(coh_avg)**2) - np.amax(np.abs(coh_avg))**2))

estNoiseFloorPerBin_dBm_cohAvg = est_noise_power_coh_avg - 10*np.log10(num_samp_range-1);
estNoiseFloorPerBin_dBm_noncohAvg = est_noise_power_non_coh_avg - 10*np.log10(num_samp_range-1);
estNoiseFloorPerNin_dBm_singleChirp = est_noise_power_range_fft - 10*np.log10(num_samp_range-1);
print('\n')
# print('\nTrue Noise Power: {0:.2f} dB \nNoise Power estimated from single chirp: {1:.2f} dB \nNoise Power non-coh Avg.: {2:.2f} dB \nNoise Power coh Avg.: {3:.2f} dB'.format(10*np.log10(noise_power), est_noise_power_range_fft,  est_noise_power_non_coh_avg, est_noise_power_coh_avg))

print('\nTrue Noise Power: {0:.2f} dB '.format(10*np.log10(noise_power)));
print('Noise Power estimated from single chirp: {0:.2f} dB '.format(est_noise_power_range_fft));
print('Noise Power non-coh Avg.: {0:.2f} dB '.format(est_noise_power_non_coh_avg))
print('Noise Power coh Avg.: {0:.2f} dB'.format(est_noise_power_coh_avg))

print('\n')
print('True Noise Floor per bin: {0:.2f} dB'.format(10*np.log10(noiseFloorPerBin)));
print('Noise Floor per bin est from single chirp: {0:.2f} dB'.format(estNoiseFloorPerNin_dBm_singleChirp))
print('Noise Floor per bin est from non-coh avg: {0:.2f} dB'.format(estNoiseFloorPerBin_dBm_noncohAvg));
print('Noise Floor per bin est from coh avg: {0:.2f} dB'.format(estNoiseFloorPerBin_dBm_cohAvg))

range_grid = np.arange(-np.pi,np.pi,2*np.pi/num_samp_range)


plt.figure(1,figsize=(20,10))
plt.plot(range_grid,20*np.log10(np.fft.fftshift(np.abs(coh_avg))),alpha=0.5)
plt.plot(range_grid,20*np.log10(np.fft.fftshift(non_coh_avg)),'k')
plt.plot(range_grid,20*np.log10(np.fft.fftshift(np.abs(signal_range_fft[:,0]))),alpha=0.5)
plt.grid(True)
plt.xlabel('omega range (rad/samp)')
plt.ylabel('Power (dB)')
plt.legend(['Coherent Avg.', 'Non-Coherent Avg.','Range FFT single chirp'])
plt.text(-0.5, -30, 'True Noise Power: '+ str(round(10*np.log10(noise_power),2)) + ' dB')
plt.text(-0.5, -37, 'Noise Power est from single chirp: '+ str(round(est_noise_power_range_fft,2)) + ' dB')
plt.text(-0.5, -44, 'Noise Power est from non-coh avg: '+ str(round(est_noise_power_non_coh_avg,2)) + ' dB')
plt.text(-0.5, -51, 'Noise Power est from coh avg: '+ str(round(est_noise_power_coh_avg,2)) + ' dB')

plt.text(-2.5, -30, 'True Noise Floor per bin: '+ str(round(10*np.log10(noiseFloorPerBin),2)) + ' dB')
plt.text(-2.5, -37, 'Noise Floor per bin est from single chirp: '+ str(round(estNoiseFloorPerNin_dBm_singleChirp,2)) + ' dB')
plt.text(-2.5, -44, 'Noise Floor per bin est from non-coh avg: '+ str(round(estNoiseFloorPerBin_dBm_noncohAvg,2)) + ' dB')
plt.text(-2.5, -51, 'Noise Floor per bin est from coh avg: '+ str(round(estNoiseFloorPerBin_dBm_cohAvg,2)) + ' dB')
