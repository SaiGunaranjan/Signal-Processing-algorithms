# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 18:01:01 2019

@author: Sai Gunaranjan Pelluri

This code can be used to generate a radar signal with the Tx and Rx arrangement as shown below:
    
    
                 
          
    *(Tx1)            * (Tx0)                       
    --------------------
    |      IC          |
    |                  |
    --------------------
    *     *     *     * 
    Rx1   Rx2   Rx3   Rx4
    

    *    *    *    *    *    *    *    * (Rx synthesized from Tx1 and Tx0)
    
    
    Elevation not modeled yet

"""

import numpy as np
import matplotlib.pyplot as plt
#import cfar_os

plt.close('all')

## System Parameters
num_fft = 1024
num_ramps = 128
num_rx = 4
num_tx= 2
adc_sampling_rate = 12e6 # 
inter_chirp_time = 62.5e-6 # micro seconds
doppler_fs = 1/inter_chirp_time
carrier_freq = 76.2e9 # GHz
speed_light = 3e8 #m/s
lamda = speed_light/carrier_freq
rx_spacing = 0.56*lamda
tx_spacing_along_x = num_rx*rx_spacing
tx_spacing_along_y = 0 #4*rx_spacing#0 #rx_spacing
az_fs = lamda/rx_spacing
chirp_sweep_bw = 2e9

# Derived system specs
range_resolution = speed_light/(2*chirp_sweep_bw)
max_range = range_resolution*num_fft
range_fs = 2*max_range
velocity_resolution = ((1/inter_chirp_time)/num_ramps)*(lamda/2)
max_velocity = (1/inter_chirp_time/2)*(lamda/2)
az_ang_resolution = np.arcsin(lamda/(num_rx*rx_spacing))*180/np.pi
max_az_angle = np.arcsin(lamda/(2*rx_spacing))*180/np.pi
#max_el_angle = np.arcsin(lamda/(2*tx_spacing_along_y))*180/np.pi


num_rx_extend = 128

# user defined parameters
num_objects = 2
object_snr = np.array([20,10])
noise_power_db = -40 # Noise Power in dB
noise_variance = 10**(noise_power_db/10)
noise_sigma = np.sqrt(noise_variance)
weights = np.sqrt(noise_variance*10**(object_snr/10))
object_range = np.array([12,33]) # range in m # np.array([9.975,19.95]) # range in m
object_velocity = np.array([11,-6]) # velocity in m/s
object_theta = np.array([-35,50]) # Theta
object_phi = np.array([0,0])


object_doppler_freq = 2*object_velocity/lamda # Hz
object_az_freq = np.sin(object_theta*np.pi/180)*np.cos(object_phi*np.pi/180)
object_el_freq = np.sin(object_theta*np.pi/180)*np.sin(object_phi*np.pi/180)
tx_x_dist = np.array([0,tx_spacing_along_x])
tx_y_dist = np.array([0,tx_spacing_along_y])
tx_phase = np.matmul(np.vstack((object_az_freq,object_el_freq)).T,np.vstack((tx_x_dist,tx_y_dist)))

range_grid = np.arange(num_fft)*range_resolution
doppler_grid = np.arange(-num_ramps//2,num_ramps//2)*velocity_resolution
angle_grid = np.arcsin(np.arange(-num_rx_extend//2,num_rx_extend//2)*lamda/(num_rx_extend*rx_spacing))*180/np.pi

radar_signal = np.zeros((2*num_fft,num_ramps,num_rx,num_tx)).astype('complex64')
for ele in np.arange(num_objects):
    range_signal = np.exp(1j*2*np.pi*object_range[ele]*np.arange(2*num_fft)/range_fs)*weights[ele]
    doppler_signal = np.exp(1j*2*np.pi*object_doppler_freq[ele]*np.arange(num_ramps)/doppler_fs)
    az_signal = np.exp(1j*2*np.pi*object_az_freq[ele]*np.arange(num_rx)/(lamda/rx_spacing))
    #el_signal = np.exp(1j*2*np.pi*object_el_freq[0]*np.arange(num_rx)/el_fs)
    #tx_phase_signal_x = np.exp(1j*2*np.pi*object_az_freq[ele]*np.arange(num_tx)/(lamda/tx_spacing_along_x))
    #tx_phase_signal_y = np.exp(1j*2*np.pi*object_el_freq[ele]*np.arange(num_tx)/(lamda/tx_spacing_along_y))
    tx_phase_signal = np.exp(1j*2*np.pi*tx_phase[ele,:]/lamda)
    radar_signal += range_signal[:,None,None,None]*doppler_signal[None,:,None,None]*az_signal[None,None,:,None]*tx_phase_signal[None,None,None,:] # [range, chirps, sensor, tx_phase]

wgn_noise = np.random.normal(0,noise_sigma/np.sqrt(2),2*num_fft*num_ramps*num_rx*num_tx) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),2*num_fft*num_ramps*num_rx*num_tx)
noise_signal = wgn_noise.reshape(2*num_fft,num_ramps,num_rx,num_tx)
radar_signal = radar_signal + noise_signal

radar_signal_pad = np.pad(radar_signal,((0,0),(0,0),(0,num_rx_extend-num_rx),(0,0)),'constant')
radar_signal_range_fft = np.fft.fft(radar_signal_pad,axis=0)[0:num_fft,:,:,:]
radar_signal_range_dopp_fft = np.fft.fft(radar_signal_range_fft,axis=1)
radar_signal_range_dopp_angle_fft = np.fft.fft(radar_signal_range_dopp_fft,axis=2)

actual_range_bins = (object_range/range_resolution).astype(int)
object_velocity_0_Fs = object_velocity.copy()
object_velocity_0_Fs[object_velocity_0_Fs<0] = object_velocity_0_Fs[object_velocity_0_Fs<0] + 2*max_velocity
actual_velocity_bins = (object_velocity_0_Fs/velocity_resolution).astype(int)


ext_virt_array = np.hstack((radar_signal_range_dopp_fft[actual_range_bins,actual_velocity_bins,0:num_rx,0],radar_signal_range_dopp_fft[actual_range_bins,actual_velocity_bins,0:num_rx,1]))
ext_virt_array_pad = np.pad(ext_virt_array,((0,0),(0,num_rx_extend-2*num_rx)),'constant') 

##object_num = 0
##plt.plot(range_grid,20*np.log10(np.abs(radar_signal_range_fft[:,:,0,0])))
##plt.plot(doppler_grid,20*np.log10(np.abs(np.fft.fftshift(radar_signal_range_dopp_fft[actual_range_bins[object_num],:,:,0],axes=0))))
##plt.plot(doppler_grid,20*np.log10(np.abs(np.fft.fftshift(radar_signal_range_dopp_fft[actual_range_bins[1],:,:,0],axes=0))))
##plt.plot(angle_grid,20*np.log10(np.abs(np.fft.fftshift(radar_signal_range_dopp_angle_fft[actual_range_bins[object_num],actual_velocity_bins[object_num],:,:],axes=0))))
##plt.plot(angle_grid,20*np.log10(np.abs(np.fft.fftshift(radar_signal_range_dopp_angle_fft[actual_range_bins[1],actual_velocity_bins[1],:,:],axes=0))))
#
#
plt.figure(1)
plt.title('Range FFT')
plt.plot(range_grid,20*np.log10(np.abs(radar_signal_range_fft[:,:,0,0])))
plt.grid(True)
plt.figure(2)
plt.title('Doppler FFT')
plt.plot(doppler_grid,20*np.log10(np.abs(np.fft.fftshift(radar_signal_range_dopp_fft[actual_range_bins[0],:,:,0],axes=0))))
plt.plot(doppler_grid,20*np.log10(np.abs(np.fft.fftshift(radar_signal_range_dopp_fft[actual_range_bins[1],:,:,0],axes=0))))
plt.grid(True)
plt.figure(3)
plt.title('Extended Sensor FFT')
plt.plot(angle_grid,20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(ext_virt_array_pad,axis=1),axes=1))).T)
plt.grid(True)

