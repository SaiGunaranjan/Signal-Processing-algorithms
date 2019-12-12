# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:14:27 2019

@author: Sai Gunaranjan Pelluri
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema


def compute_Twolargest_localpeaks(x):

    Ncases= x.shape[0]
    
    twoLargestLocalPeaks_matrix = np.zeros((0,2)).astype('int64')
    
    for i in range(Ncases):
        data=x[i,:]
        inx= argrelextrema(data,np.greater,order=2)[0]
        twoLargestLocalPeaks = inx[np.argsort(data[inx])][-2::]
        twoLargestLocalPeaks_matrix = np.vstack((twoLargestLocalPeaks,twoLargestLocalPeaks_matrix))
        
    return twoLargestLocalPeaks_matrix


plt.close('all');
num_chirps_tx1 = 128;
num_chirps_tx2 = 10;
num_chirps_tx3 = 19;
num_fft = 256;
object_snr = np.array([0,0]);
freq_ind_sep = 20
freq_ind_1 = np.random.randint(0,num_chirps_tx1-1);
freq_ind_2 = freq_ind_1 + freq_ind_sep;
freq_ind = np.array([freq_ind_1,freq_ind_2])
dig_freq = (2*np.pi*freq_ind)/num_chirps_tx1;

num_objects = 2;

noise_power_db = -40; # Noise Power in dB
noise_variance = 10**(noise_power_db/10);
noise_sigma = np.sqrt(noise_variance);
signal_power = noise_variance*10**(object_snr/10);
signal_amp_real = np.sqrt(signal_power);
random_phase = np.exp(1j*2*np.pi*np.random.uniform(0,1,num_objects));
true_complex_weights = signal_amp_real*random_phase

signal_tx1 = np.sum(np.exp(1j*np.matmul(dig_freq[:,None],np.arange(num_chirps_tx1)[None,:]))*true_complex_weights[:,None],axis=0);
noise_tx1 = np.random.normal(0,noise_sigma/np.sqrt(2),num_chirps_tx1) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),num_chirps_tx1);
signal_noise_tx1 = signal_tx1 + noise_tx1;
signal_noise_tx1_fft = np.fft.fft(signal_noise_tx1,n=num_fft)/num_chirps_tx1;
signal_noise_tx1_spectrum = 20*np.log10(np.abs(signal_noise_tx1_fft));
#est_freq_ind = np.argsort(signal_noise_tx1_spectrum)[-num_objects::]
est_freq_ind = compute_Twolargest_localpeaks(signal_noise_tx1_spectrum[None,:])[0]
est_dig_freq = (2*np.pi*est_freq_ind)/num_fft
est_complex_weights_tx1_fft = signal_noise_tx1_fft[est_freq_ind];



noise_tx2 = np.random.normal(0,noise_sigma/np.sqrt(2),num_chirps_tx2) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),num_chirps_tx2);
signal_tx2 = signal_tx1[0:num_chirps_tx2];
signal_noise_tx2 = signal_tx2 + noise_tx2;
vandermone_matrix = np.exp(1j*np.matmul(np.arange(num_chirps_tx2)[:,None],est_dig_freq[None,:]));
est_complex_weights_tx2 = np.matmul(np.linalg.pinv(vandermone_matrix),signal_noise_tx2);
signal_noise_tx2_fft = np.fft.fft(signal_noise_tx2,n=num_fft)/num_chirps_tx2;
signal_noise_tx2_spectrum = 20*np.log10(np.abs(signal_noise_tx2_fft));

noise_tx3 = np.random.normal(0,noise_sigma/np.sqrt(2),num_chirps_tx3) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),num_chirps_tx3);
signal_tx3 = signal_tx1[0:num_chirps_tx3]
signal_noise_tx3 = signal_tx3 + noise_tx3;
signal_noise_tx3_fft = np.fft.fft(signal_noise_tx3,n=num_fft)/num_chirps_tx3;
signal_noise_tx3_spectrum = 20*np.log10(np.abs(signal_noise_tx3_fft));
est_freq_ind_tx3 = compute_Twolargest_localpeaks(signal_noise_tx3_spectrum[None,:])[0]
est_dig_freq_tx3 = (2*np.pi*est_freq_ind_tx3)/num_fft
est_complex_weights_tx3_fft = signal_noise_tx3_fft[est_freq_ind];

print('True amplitude (dB)',20*np.log10(np.abs(true_complex_weights)), 'True phase (deg)', np.angle(true_complex_weights)*180/np.pi);                       
print('Tx1 FFT based Est amplitude (dB)',20*np.log10(np.abs(est_complex_weights_tx1_fft)), 'Tx1 FFT based Est phase (deg)', np.angle(est_complex_weights_tx1_fft)*180/np.pi);
print('Tx2 LS based Est amplitude(dB)',20*np.log10(np.abs(est_complex_weights_tx2)), 'Tx2 LS based Est phase (deg)', np.angle(est_complex_weights_tx2)*180/np.pi);
print('Tx3 FFT based Est amplitude(dB)',20*np.log10(np.abs(est_complex_weights_tx3_fft)), 'Tx3 FFT based Est phase (deg)',np.angle(est_complex_weights_tx3_fft)*180/np.pi)


plt.figure(1,figsize=(16,9))
plt.plot(signal_noise_tx1_spectrum,'-o',label='Tx1 spectrum');
plt.plot(signal_noise_tx2_spectrum,'-o',label='Tx2 spectrum');
plt.plot(signal_noise_tx3_spectrum,'-o',label='Tx3 spectrum');
plt.xlabel('Frequency Index');
plt.ylabel('Power (dB)');
plt.legend()
plt.grid(True)
