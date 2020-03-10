# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 15:30:40 2019

@author: Sai Gunaranjan Pelluri
"""
####################################CFAR-OS###################################################
# FFT_SignalVector:Give Range FFT or Doppler FFT as your signal vector
# GuardBandLength : To discard some of the adjacent samples surrounding the Cell under test
# CFAR_Window_Length: Sliding window length 
# Threshold_Beta : Linear multiplicative factor
# OrderedStatisticIndex:Pick the Kth index peak out of the window values

import numpy as np
import matplotlib.pyplot as plt
import cfar_lib

 




if 0:
    plt.close('all')
    num_fft = 1024
    n_vec = np.arange(2*num_fft)
    num_bjs = 5
    freq_vec = np.array([20,150,300,350,400])
    object_snr = np.array([15,3,12,10,5])
    noise_power_db = -30 # Noise Power in dB
    noise_variance = 10**(noise_power_db/10)
    noise_sigma = np.sqrt(noise_variance)
    weights = noise_variance*10**(object_snr/10)
    fs = 1e3
    radar_signal = np.zeros((2*num_fft)).astype('complex64')
    for ele in np.arange(num_bjs):
        radar_signal += weights[ele]*np.exp(1j*2*np.pi*freq_vec[ele]*n_vec/fs)
    wgn_noise = np.random.normal(0,noise_sigma/np.sqrt(2),2*num_fft) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),2*num_fft)
    radar_signal = radar_signal + wgn_noise
    radar_signal_fft = np.fft.fft(radar_signal)
    FFT_SignalVector = radar_signal_fft[0:num_fft]
    freq_grid = np.arange(num_fft)*fs/(2*num_fft)
    signal_mag = np.abs(FFT_SignalVector)**2
    GuardBandLength = 1
    valid_samp_len = 18
    false_alarm_rate = 1e-5
    OrderedStatisticIndex = 3 # 
    
    bool_array = cfar_lib.CFAR_OS(signal_mag,GuardBandLength,valid_samp_len,false_alarm_rate,OrderedStatisticIndex)
    det_indices = np.where(bool_array>0)
    det_freq = det_indices[0]*fs/(2*num_fft)
    print('True frequencies: ', freq_vec, 'Estimated frequencies: ', det_freq)
    plt.figure(1)
    plt.title('CFAR OS')
    plt.plot(freq_grid,20*np.log10(np.abs(FFT_SignalVector)))
    plt.plot(det_freq,20*np.log10(np.abs(FFT_SignalVector[det_indices[0]])),'rD')
    plt.xlabel('Frequency in Hz')
    plt.ylabel('Power (dB)')
    plt.grid(True)
    
    bool_array = cfar_lib.CFAR_CA(signal_mag,GuardBandLength,valid_samp_len,false_alarm_rate)
    det_indices = np.where(bool_array>0)
    det_freq = det_indices[0]*fs/(2*num_fft)
    print('True frequencies: ', freq_vec, 'Estimated frequencies: ', det_freq)
    plt.figure(2)
    plt.title('CFAR CA sai')
    plt.plot(freq_grid,20*np.log10(np.abs(FFT_SignalVector)))
    plt.plot(det_freq,20*np.log10(np.abs(FFT_SignalVector[det_indices[0]])),'rD')
    plt.xlabel('Frequency in Hz')
    plt.ylabel('Power (dB)')
    plt.grid(True)
    
    



if 0:
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
    radar_signal_range_fft = np.fft.fft(radar_signal,axis=0)
    radar_signal_range_fft_dopp_fft = np.fft.fft(radar_signal_range_fft,axis=1)
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
    
    bool_array = cfar_lib.CFAR_OS_2D(signal_mag, guardband_len_x, guardband_len_y, valid_samp_len_x, valid_samp_len_y, false_alarm_rate, OrderedStatisticIndex)
    det_indices = np.where(bool_array>0)
    plt.figure(1)
    plt.title('CFAR OS 2D')
    plt.imshow(10*np.log10(signal_mag),aspect='auto')
    plt.scatter(det_indices[1],det_indices[0],c='r',marker='*',s=20)
    print('True range bins:',range_bins_ind, ', CFAR det range bins:', det_indices[0])
    print('True doppler bins:',doppler_bins_ind, ', CFAR det doppler bins:', det_indices[1])
    
    bool_array = cfar_lib.CFAR_CA_2D(signal_mag, guardband_len_x, guardband_len_y, valid_samp_len_x, valid_samp_len_y, false_alarm_rate)
    det_indices = np.where(bool_array>0)
    plt.figure(2)
    plt.title('CFAR CA 2D')
    plt.imshow(10*np.log10(signal_mag),aspect='auto')
    plt.scatter(det_indices[1],det_indices[0],c='r',marker='*',s=20)
    print('True range bins:',range_bins_ind, ', CFAR det range bins:', det_indices[0])
    print('True doppler bins:',doppler_bins_ind, ', CFAR det doppler bins:', det_indices[1])
    
    bool_array = cfar_lib.CFAR_CA_2D_cross(signal_mag, guardband_len_x, guardband_len_y, valid_samp_len_x, valid_samp_len_y, false_alarm_rate)
    det_indices = np.where(bool_array>0)
    plt.figure(3)
    plt.title('CFAR CA 2D cross')
    plt.imshow(10*np.log10(signal_mag),aspect='auto')
    plt.scatter(det_indices[1],det_indices[0],c='r',marker='*',s=20)
    print('True range bins:',range_bins_ind, ', CFAR det range bins:', det_indices[0])
    print('True doppler bins:',doppler_bins_ind, ', CFAR det doppler bins:', det_indices[1])

    
if 0:
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
    object_snr = np.array([20,20,20,20])
    noise_power_db = -20 # Noise Power in dB
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
    radar_signal_range_fft = np.fft.fft(radar_signal,axis=0)
    radar_signal_range_fft_dopp_fft = np.fft.fft(radar_signal_range_fft,axis=1)
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
    
    
    noise_map = cfar_lib.CFAR_CA_2D_cross_map(signal_mag, guardband_len_x, guardband_len_y, valid_samp_len_x, valid_samp_len_y)
    plt.figure(3)
    plt.subplot(121)
    plt.title('True Range Doppler Image')
    plt.imshow(10*np.log10(signal_mag),aspect='auto')
    plt.colorbar()
    plt.subplot(122)
    plt.title('CFAR CA avged noise map')
    plt.imshow(10*np.log10(noise_map),aspect='auto')
    plt.colorbar()

    

