# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 20:01:56 2020

@author: saiguna
"""


import numpy as np
import matplotlib.pyplot as plt
import cfar_lib


plt.close('all')
num_fft = 512#1024
num_ramps = 512
num_objs = 4
fs_range = 1e3
fs_dopp = 1e3
range_freq_vec = np.array([-200,200,350,400])
doppler_freq_vec = np.array([-50,250,300,450])

range_freq_grid = np.arange(-num_fft//2,num_fft//2)*fs_range/(num_fft)
dopp_freq_grid = np.arange(-num_ramps//2,num_ramps//2)*fs_dopp/(num_ramps)

#object_snr = np.array([15,12,10,5])
object_snr = np.array([0,20,20,20])
noise_power_db = -30 # Noise Power in dB
noise_variance = 10**(noise_power_db/10)
noise_sigma = np.sqrt(noise_variance)
weights = noise_variance*10**(object_snr/10)

radar_signal = np.zeros((num_fft,num_ramps)).astype('complex64')
for ele in np.arange(num_objs):
    range_signal = weights[ele]*np.exp(1j*2*np.pi*range_freq_vec[ele]*np.arange(num_fft)/fs_range)
    doppler_signal = np.exp(1j*2*np.pi*doppler_freq_vec[ele]*np.arange(num_ramps)/fs_dopp)
    radar_signal += range_signal[:,None]*doppler_signal[None,:] # [range,num_ramps]
    
    
wgn_noise = np.random.normal(0,noise_sigma/np.sqrt(2),num_fft*num_ramps) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),num_fft*num_ramps)
noise_signal = wgn_noise.reshape(num_fft,num_ramps)
radar_signal = radar_signal + noise_signal
radar_signal_range_fft = np.fft.fft(radar_signal,axis=0)/num_fft
radar_signal_range_fft_dopp_fft = np.fft.fft(radar_signal_range_fft,axis=1)/num_ramps
range_freq_vec[range_freq_vec<0] = range_freq_vec[range_freq_vec<0] + fs_range
range_bins_ind = (range_freq_vec/(fs_range/num_fft)).astype(int)
doppler_freq_vec[doppler_freq_vec<0] = doppler_freq_vec[doppler_freq_vec<0] + fs_dopp
doppler_bins_ind = (doppler_freq_vec/(fs_dopp/num_ramps)).astype(int)
signal_mag = np.abs(radar_signal_range_fft_dopp_fft)**2
guardband_len_x = 1
guardband_len_y = 1
valid_samp_len_x = 20
valid_samp_len_y = 20
false_alarm_rate = 1e-5#1e-4
OrderedStatisticIndex = 3 # 

print('True range bins:',range_bins_ind)
print('True Doppler bins:', doppler_bins_ind,'\n')

bool_array_os = cfar_lib.CFAR_OS_2D(signal_mag, guardband_len_x, guardband_len_y, valid_samp_len_x, valid_samp_len_y, false_alarm_rate, OrderedStatisticIndex)
det_indices_os = np.where(bool_array_os>0)
print('CFAR OS det range bins:', det_indices_os[0])
print('CFAR OS det doppler bins:', det_indices_os[1],'\n')

bool_array_osCross = cfar_lib.CFAR_OS_2D(signal_mag, guardband_len_x, guardband_len_y, valid_samp_len_x, valid_samp_len_y, false_alarm_rate, OrderedStatisticIndex)
det_indices_osCross = np.where(bool_array_osCross>0)
print('CFAR OS cross det range bins:', det_indices_osCross[0])
print('CFAR OS cross det doppler bins:', det_indices_osCross[1],'\n')

bool_array_ca = cfar_lib.CFAR_CA_2D(signal_mag, guardband_len_x, guardband_len_y, valid_samp_len_x, valid_samp_len_y, false_alarm_rate)
det_indices_ca = np.where(bool_array_ca>0)
print('CFAR CA det range bins:', det_indices_ca[0])
print('CFAR CA det doppler bins:', det_indices_ca[1],'\n')

bool_array_caCross = cfar_lib.CFAR_CA_2D_cross(signal_mag, guardband_len_x, guardband_len_y, valid_samp_len_x, valid_samp_len_y, false_alarm_rate)
det_indices_caCross = np.where(bool_array_caCross>0)
print('CFAR CA cross det range bins:', det_indices_caCross[0])
print('CFAR CA cross det doppler bins:', det_indices_caCross[1],'\n')

noise_map = cfar_lib.CFAR_CA_2D_cross_map(signal_mag, guardband_len_x, guardband_len_y, valid_samp_len_x, valid_samp_len_y)



plt.figure(1,figsize=(20,10))

plt.subplot(2,2,1)
plt.title('CFAR OS 2D')
plt.imshow(10*np.log10(signal_mag),aspect='auto')
plt.scatter(det_indices_os[1],det_indices_os[0],c='r',marker='*',s=20)
plt.ylabel('Range Index')
plt.xlabel('Doppler Index')


plt.subplot(2,2,2)
plt.title('CFAR OS 2D Cross')
plt.imshow(10*np.log10(signal_mag),aspect='auto')
plt.scatter(det_indices_osCross[1],det_indices_osCross[0],c='r',marker='*',s=20)
plt.ylabel('Range Index')
plt.xlabel('Doppler Index')


plt.subplot(2,2,3)
plt.title('CFAR CA 2D')
plt.imshow(10*np.log10(signal_mag),aspect='auto')
plt.scatter(det_indices_ca[1],det_indices_ca[0],c='r',marker='*',s=20)
plt.ylabel('Range Index')
plt.xlabel('Doppler Index')


plt.subplot(2,2,4)
plt.title('CFAR CA 2D cross')
plt.imshow(10*np.log10(signal_mag),aspect='auto');
plt.scatter(det_indices_caCross[1],det_indices_caCross[0],c='r',marker='*',s=20)
plt.ylabel('Range Index')
plt.xlabel('Doppler Index')



plt.figure(3,figsize=(20,10))
plt.subplot(121)
plt.title('True Range Doppler Image')
plt.imshow(10*np.log10(signal_mag),aspect='auto')
plt.ylabel('Range Index')
plt.xlabel('Doppler Index')
plt.colorbar()
plt.subplot(122)
plt.title('CFAR CA avged noise map')
plt.imshow(10*np.log10(noise_map),aspect='auto')
plt.ylabel('Range Index')
plt.xlabel('Doppler Index')
plt.colorbar()