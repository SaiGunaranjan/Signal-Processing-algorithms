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


freq_ind_1 = 10#np.random.randint(0,num_chirps_tx1-1);

num_objects = 2;

noise_power_db = -40; # Noise Power in dB
noise_variance = 10**(noise_power_db/10);
noise_sigma = np.sqrt(noise_variance);

bin_sep = np.arange(2,14,3);
SNR = np.arange(-20,30,2);
num_montecarlo = 100;

rmse_amp_tx1_snr_binspace = np.zeros((SNR.shape[0],num_objects,0)).astype('float32');
rmse_amp_tx2_snr_binspace = np.zeros((SNR.shape[0],num_objects,0)).astype('float32');
rmse_amp_tx3_snr_binspace = np.zeros((SNR.shape[0],num_objects,0)).astype('float32');
rmse_ang_tx1_snr_binspace = np.zeros((SNR.shape[0],num_objects,0)).astype('float32');
rmse_ang_tx2_snr_binspace = np.zeros((SNR.shape[0],num_objects,0)).astype('float32');
rmse_ang_tx3_snr_binspace = np.zeros((SNR.shape[0],num_objects,0)).astype('float32');

for freq_ind_sep in bin_sep:
    freq_ind_perturb = np.random.uniform(-0.5,0.5,1); #np.random.uniform(-0.5,0.5,1);
    freq_ind_1_perturb = freq_ind_1 + 0*freq_ind_perturb;
    freq_ind_2 = freq_ind_1_perturb + freq_ind_sep;
    freq_ind = np.array([freq_ind_1_perturb,freq_ind_2])
    dig_freq = (2*np.pi*freq_ind)/num_chirps_tx1;
    rmse_amp_tx1_snr = np.zeros((0,num_objects)).astype('float32');
    rmse_amp_tx2_snr = np.zeros((0,num_objects)).astype('float32');
    rmse_amp_tx3_snr = np.zeros((0,num_objects)).astype('float32');
    rmse_ang_tx1_snr = np.zeros((0,num_objects)).astype('float32');
    rmse_ang_tx2_snr = np.zeros((0,num_objects)).astype('float32');
    rmse_ang_tx3_snr = np.zeros((0,num_objects)).astype('float32');
    
    for snr in SNR:
        object_snr = np.array([snr,snr]);
        signal_power = noise_variance*10**(object_snr/10);
        signal_amp_real = np.sqrt(signal_power);
        random_phase = 2*np.pi*np.random.uniform(-0.5,0.5,num_objects)
        random_phasor = np.exp(1j*random_phase);
        true_complex_weights = signal_amp_real*random_phasor
        signal_tx1 = np.sum(np.exp(1j*np.matmul(dig_freq,np.arange(num_chirps_tx1)[None,:]))*true_complex_weights[:,None],axis=0);
        
        signal_tx2 = signal_tx1[0:num_chirps_tx2];
        signal_tx3 = signal_tx1[0:num_chirps_tx3];
        
        errorMat_amp_tx1 = np.zeros((0,num_objects)).astype('float32');
        errorMat_amp_tx2 = np.zeros((0,num_objects)).astype('float32');
        errorMat_amp_tx3 = np.zeros((0,num_objects)).astype('float32');
        errorMat_ang_tx1 = np.zeros((0,num_objects)).astype('float32');
        errorMat_ang_tx2 = np.zeros((0,num_objects)).astype('float32');
        errorMat_ang_tx3 = np.zeros((0,num_objects)).astype('float32');
        
        for iter_num in np.arange(num_montecarlo):
        
            noise_tx1 = np.random.normal(0,noise_sigma/np.sqrt(2),num_chirps_tx1) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),num_chirps_tx1);
            signal_noise_tx1 = signal_tx1 + noise_tx1;
            signal_noise_tx1_fft = np.fft.fft(signal_noise_tx1,n=num_fft)/num_chirps_tx1;
            signal_noise_tx1_spectrum = 20*np.log10(np.abs(signal_noise_tx1_fft));
            est_freq_ind = compute_Twolargest_localpeaks(signal_noise_tx1_spectrum[None,:])[0]
            est_dig_freq = (2*np.pi*est_freq_ind)/num_fft
            est_complex_weights_tx1_fft = signal_noise_tx1_fft[est_freq_ind];
            
            
            
            noise_tx2 = np.random.normal(0,noise_sigma/np.sqrt(2),num_chirps_tx2) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),num_chirps_tx2);
            signal_noise_tx2 = signal_tx2 + noise_tx2;
            vandermone_matrix = np.exp(1j*np.matmul(np.arange(num_chirps_tx2)[:,None],est_dig_freq[None,:]));
            est_complex_weights_tx2 = np.matmul(np.linalg.pinv(vandermone_matrix),signal_noise_tx2);

            
            noise_tx3 = np.random.normal(0,noise_sigma/np.sqrt(2),num_chirps_tx3) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),num_chirps_tx3);
            signal_noise_tx3 = signal_tx3 + noise_tx3;
            signal_noise_tx3_fft = np.fft.fft(signal_noise_tx3,n=num_fft)/num_chirps_tx3;
            signal_noise_tx3_spectrum = 20*np.log10(np.abs(signal_noise_tx3_fft));
            est_freq_ind_tx3 = compute_Twolargest_localpeaks(signal_noise_tx3_spectrum[None,:])[0]
            est_dig_freq_tx3 = (2*np.pi*est_freq_ind_tx3)/num_fft;
            est_complex_weights_tx3_fft = signal_noise_tx3_fft[est_freq_ind_tx3];
            
            error_amp_tx1 = 20*np.log10(np.sort(signal_amp_real))- 20*np.log10(np.sort(np.abs(est_complex_weights_tx1_fft)));
            errorMat_amp_tx1 = np.vstack((errorMat_amp_tx1,error_amp_tx1));
            error_amp_tx2 = 20*np.log10(np.sort(signal_amp_real))- 20*np.log10(np.sort(np.abs(est_complex_weights_tx2)));
            errorMat_amp_tx2 = np.vstack((errorMat_amp_tx2,error_amp_tx2));
            error_amp_tx3 = 20*np.log10(np.sort(signal_amp_real))-20*np.log10(np.sort(np.abs(est_complex_weights_tx3_fft)));
            errorMat_amp_tx3 = np.vstack((errorMat_amp_tx3,error_amp_tx3));
            
            
            error_ang_tx1 = (np.sort(random_phase)-np.sort(np.angle(est_complex_weights_tx1_fft)))*180/np.pi;
            errorMat_ang_tx1 = np.vstack((errorMat_ang_tx1,error_ang_tx1));
            error_ang_tx2 = (np.sort(random_phase)-np.sort(np.angle(est_complex_weights_tx2)))*180/np.pi;
            errorMat_ang_tx2 = np.vstack((errorMat_ang_tx2,error_ang_tx2));
            error_ang_tx3 = (np.sort(random_phase)-np.sort(np.angle(est_complex_weights_tx3_fft)))*180/np.pi;
            errorMat_ang_tx3 = np.vstack((errorMat_ang_tx3,error_ang_tx3));


            
        rmse_amp_tx1 = np.sqrt(np.sum(errorMat_amp_tx1**2,axis=0)/(num_montecarlo));
        rmse_amp_tx2 = np.sqrt(np.sum(errorMat_amp_tx2**2,axis=0)/(num_montecarlo));
        rmse_amp_tx3 = np.sqrt(np.sum(errorMat_amp_tx3**2,axis=0)/(num_montecarlo));
        
        rmse_ang_tx1 = np.sqrt(np.sum(errorMat_ang_tx1**2,axis=0)/(num_montecarlo));
        rmse_ang_tx2 = np.sqrt(np.sum(errorMat_ang_tx2**2,axis=0)/(num_montecarlo));
        rmse_ang_tx3 = np.sqrt(np.sum(errorMat_ang_tx3**2,axis=0)/(num_montecarlo));
        
        
        rmse_amp_tx1_snr = np.vstack((rmse_amp_tx1_snr,rmse_amp_tx1));
        rmse_amp_tx2_snr = np.vstack((rmse_amp_tx2_snr,rmse_amp_tx2));
        rmse_amp_tx3_snr = np.vstack((rmse_amp_tx3_snr,rmse_amp_tx3));
        
        rmse_ang_tx1_snr = np.vstack((rmse_ang_tx1_snr,rmse_ang_tx1));
        rmse_ang_tx2_snr = np.vstack((rmse_ang_tx2_snr,rmse_ang_tx2));
        rmse_ang_tx3_snr = np.vstack((rmse_ang_tx3_snr,rmse_ang_tx3));
        
        
    rmse_amp_tx1_snr_binspace = np.dstack((rmse_amp_tx1_snr_binspace,rmse_amp_tx1_snr));
    rmse_amp_tx2_snr_binspace = np.dstack((rmse_amp_tx2_snr_binspace,rmse_amp_tx2_snr));
    rmse_amp_tx3_snr_binspace = np.dstack((rmse_amp_tx3_snr_binspace,rmse_amp_tx3_snr));
    
    rmse_ang_tx1_snr_binspace = np.dstack((rmse_ang_tx1_snr_binspace,rmse_ang_tx1_snr));
    rmse_ang_tx2_snr_binspace = np.dstack((rmse_ang_tx2_snr_binspace,rmse_ang_tx2_snr));
    rmse_ang_tx3_snr_binspace = np.dstack((rmse_ang_tx3_snr_binspace,rmse_ang_tx3_snr));



bin_sep_legend = ['bin_sep = ' + str(ele) for ele in bin_sep];


#plt.figure(1,figsize=(16,9));
#plt.subplot(1,2,1);
#plt.title('Amplitude RMSE of 1st object from 128 chirps');
#plt.plot(SNR,rmse_amp_tx1_snr_binspace[:,0,:],'-o');
#plt.xlabel('SNR (dB)');
#plt.ylabel('RMSE (dB)');
#plt.legend(bin_sep_legend);
#plt.grid(True);
#plt.subplot(1,2,2);
#plt.title('Amplitude RMSE of 2nd object from 128 chirps');
#plt.plot(SNR,rmse_amp_tx1_snr_binspace[:,1,:],'-o');
#plt.xlabel('SNR (dB)');
#plt.ylabel('RMSE (dB)');
#plt.legend(bin_sep_legend);
#plt.grid(True);

plt.figure(2,figsize=(16,9));
plt.subplot(1,2,1);
plt.title('Angle RMSE of 1st object from 128 chirps');
plt.plot(SNR,rmse_ang_tx1_snr_binspace[:,0,:],'-o');
plt.xlabel('SNR (dB)');
plt.ylabel('RMSE (deg)');
plt.grid(True);
plt.legend(bin_sep_legend);
plt.subplot(1,2,2);
plt.title('Angle RMSE of 1st object from 10 chirp using LS');
plt.plot(SNR,rmse_ang_tx2_snr_binspace[:,0,:],'-o');
plt.xlabel('SNR (dB)');
plt.ylabel('RMSE (deg)');
plt.grid(True);
plt.legend(bin_sep_legend);






#plt.figure(3,figsize=(16,9));
#plt.subplot(1,2,1);
#plt.title('Amplitude RMSE of 1st object from 10 chirps');
#plt.plot(SNR,rmse_amp_tx2_snr_binspace[:,0,:],'-o');
#plt.xlabel('SNR (dB)');
#plt.ylabel('RMSE (dB)');
#plt.legend(bin_sep_legend);
#plt.grid(True);
#plt.subplot(1,2,2);
#plt.title('Amplitude RMSE of 2nd object from 10 chirps');
#plt.plot(SNR,rmse_amp_tx2_snr_binspace[:,1,:],'-o');
#plt.xlabel('SNR (dB)');
#plt.ylabel('RMSE (dB)');
#plt.legend(bin_sep_legend);
#plt.grid(True);

plt.figure(4,figsize=(16,9));
plt.subplot(1,2,1);
plt.title('Angle RMSE of 2nd object from 128 chirps');
plt.plot(SNR,rmse_ang_tx1_snr_binspace[:,1,:],'-o');
plt.xlabel('SNR (dB)');
plt.ylabel('RMSE (deg)');
plt.grid(True);
plt.legend(bin_sep_legend);
plt.subplot(1,2,2);
plt.title('Angle RMSE of 2nd object from 10 chirps using LS');
plt.plot(SNR,rmse_ang_tx2_snr_binspace[:,1,:],'-o');
plt.xlabel('SNR (dB)');
plt.ylabel('RMSE (deg)');
plt.grid(True);
plt.legend(bin_sep_legend);




#plt.figure(5,figsize=(16,9));
#plt.subplot(1,2,1);
#plt.title('Amplitude RMSE of 1st object from 19 chirps');
#plt.plot(SNR,rmse_amp_tx3_snr_binspace[:,0,:],'-o');
#plt.xlabel('SNR (dB)');
#plt.ylabel('RMSE (dB)');
#plt.legend(bin_sep_legend);
#plt.grid(True);
#plt.subplot(1,2,2);
#plt.title('Amplitude RMSE of 2nd object from 19 chirps');
#plt.plot(SNR,rmse_amp_tx3_snr_binspace[:,1,:],'-o');
#plt.xlabel('SNR (dB)');
#plt.ylabel('RMSE (dB)');
#plt.legend(bin_sep_legend);
#plt.grid(True);

#plt.figure(6,figsize=(16,9));
#plt.subplot(1,2,1);
#plt.title('Angle RMSE of 1st object from 19 chirp');
#plt.plot(SNR,rmse_ang_tx3_snr_binspace[:,0,:],'-o');
#plt.xlabel('SNR (dB)');
#plt.ylabel('RMSE (deg)');
#plt.grid(True);
#plt.legend(bin_sep_legend);
#plt.subplot(1,2,2);
#plt.title('Angle RMSE of 2nd object from 19 chirps');
#plt.plot(SNR,rmse_ang_tx3_snr_binspace[:,1,:],'-o');
#plt.xlabel('SNR (dB)');
#plt.ylabel('RMSE (deg)');
#plt.grid(True);
#plt.legend(bin_sep_legend);