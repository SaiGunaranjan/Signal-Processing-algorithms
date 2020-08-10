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
num_chirps_tx1 = 100;
num_chirps_tx2 = 10;
num_fft = 100#128;


freq_ind_1 = 10#np.random.randint(0,num_chirps_tx1-1);

num_objects = 2;

noise_power_db = -40; # Noise Power in dB
noiseFloorPerBin_dB = -70 #noise_power_db - 10*np.log10(num_chirps_tx1) + 27
noise_variance = 10**((noiseFloorPerBin_dB - 10)/10);
noise_sigma = np.sqrt(noise_variance);

bin_sep = np.arange(0.1,40,0.1)
bin_sep_ls_samples = (num_chirps_tx2/num_chirps_tx1)*bin_sep
SNR = np.arange(-9,14,3); 
num_binsep = len(bin_sep)
num_snr = len(SNR)
num_montecarlo = 100;


ampError_tx2 = np.zeros((num_montecarlo,num_objects,num_snr,num_binsep))
angleError_tx2 = np.zeros((num_montecarlo,num_objects,num_snr,num_binsep))
absError_tx2 = np.zeros((num_montecarlo,num_objects,num_snr,num_binsep))

cnt_bin = 0
for freq_ind_sep in bin_sep:
    freq_ind_perturb = np.random.uniform(-0.5,0.5,1); #np.random.uniform(-0.5,0.5,1);
    freq_ind_1_perturb = freq_ind_1 + freq_ind_perturb;
    freq_ind_2 = freq_ind_1_perturb + freq_ind_sep;
    freq_ind = np.array([freq_ind_1_perturb,freq_ind_2])
    dig_freq = (2*np.pi*freq_ind)/num_chirps_tx1;
    
    
    cnt_snr = 0
    for snr in SNR:
        object_snr = np.array([snr,snr]);
        signal_amp_real = 10**((noiseFloorPerBin_dB + object_snr - 10)/20)*2
    
        cnt_mc = 0
        for iter_num in np.arange(num_montecarlo):
            
            random_phase = np.hstack((0,2*np.pi*np.random.uniform(-0.5,0.5,1)))
            random_phasor = np.exp(1j*random_phase);
            true_complex_weights = signal_amp_real*random_phasor
            signal_tx1 = np.sum(np.exp(1j*np.matmul(dig_freq,np.arange(num_chirps_tx1)[None,:]))*true_complex_weights[:,None],axis=0);
            
            signal_tx2 = signal_tx1[0:num_chirps_tx2];
           
        
            # noise_tx1 = np.random.normal(0,noise_sigma/np.sqrt(2),num_chirps_tx1) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),num_chirps_tx1);
            noise_tx1 = noise_sigma*np.sqrt(num_chirps_tx1)*(np.random.randn(num_chirps_tx1) + 1j*np.random.randn(num_chirps_tx1))
            signal_noise_tx1 = signal_tx1 + noise_tx1;
            signal_noise_tx1_fft = np.fft.fft(np.hamming(num_chirps_tx1)*signal_noise_tx1,n=num_fft)/num_chirps_tx1;
            signal_noise_tx1_spectrum = 20*np.log10(np.abs(signal_noise_tx1_fft));
            est_freq_ind = compute_Twolargest_localpeaks(signal_noise_tx1_spectrum[None,:])[0]
            est_freq_ind = np.sort(est_freq_ind);
            est_dig_freq = (2*np.pi*est_freq_ind)/num_fft
            est_complex_weights_tx1_fft = signal_noise_tx1_fft[est_freq_ind];
            
            
            
            # noise_tx2 = np.random.normal(0,noise_sigma/np.sqrt(2),num_chirps_tx2) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),num_chirps_tx2);
            noise_tx2 = noise_sigma*np.sqrt(num_chirps_tx2)*(np.random.randn(num_chirps_tx2) + 1j*np.random.randn(num_chirps_tx2))
            signal_noise_tx2 = signal_tx2 + noise_tx2;
            signal_noise_tx2_fft = np.fft.fft(np.hamming(num_chirps_tx2)*signal_noise_tx2,n=num_fft)/num_chirps_tx2;
            # vandermond_matrix = np.exp(1j*np.matmul(np.arange(num_chirps_tx2)[:,None],est_dig_freq[None,:]));
            vandermond_matrix = np.exp(1j*np.matmul(np.arange(num_chirps_tx2)[:,None],dig_freq.T));
            est_complex_weights_tx2 = np.matmul(np.linalg.pinv(vandermond_matrix),signal_noise_tx2);
            
            
            error_amp_tx2 = np.abs(true_complex_weights) - np.abs(est_complex_weights_tx2);
            error_angle_tx2 = np.angle(true_complex_weights) - np.angle(est_complex_weights_tx2)
            error_abs_tx2 = np.abs((true_complex_weights - est_complex_weights_tx2)/true_complex_weights);
            
            
            
            ampError_tx2[cnt_mc,:,cnt_snr,cnt_bin] = error_amp_tx2;
            angleError_tx2[cnt_mc,:,cnt_snr,cnt_bin] = error_angle_tx2;
            absError_tx2[cnt_mc,:,cnt_snr,cnt_bin] = error_abs_tx2
            
            cnt_mc += 1;
            
        cnt_snr += 1;
    
    cnt_bin += 1



snr_legend = ['input SNR (dB) = ' + str(ele) for ele in SNR];

rmse_error_abs_tx2 = np.std(absError_tx2,axis=0)
percentile_error_abs_tx2 = np.percentile(absError_tx2,99,axis=0)
# rmse_error_abs_tx2 = np.sqrt(np.sum(absError_tx2**2,axis=0)/(num_montecarlo));
rmse_error_abs_tx2_dB = 20*np.log10(rmse_error_abs_tx2[0,:,:])
percentile_error_abs_tx2_dB = 20*np.log10(percentile_error_abs_tx2)


rmse_error_amp_tx2 = 20*np.log10(np.sqrt(np.sum(ampError_tx2**2,axis=0)/(num_montecarlo)));
rmse_error_angle_tx2 = np.sqrt(np.sum(angleError_tx2**2,axis=0)/(num_montecarlo));
rmse_error_angle_tx2_dB = 20*np.log10(np.sqrt(np.sum(angleError_tx2**2,axis=0)/(num_montecarlo)));

kneePoint = 0.6
decade_kneePoint = kneePoint/10
xaxis_pts = np.array([decade_kneePoint,kneePoint])

plt.figure(7,figsize=(20,10));
plt.title('Fractional error (dB)');
plt.plot(bin_sep_ls_samples,rmse_error_abs_tx2_dB.T,'-o');
plt.xscale('log');
plt.axvline(kneePoint, color="black", linestyle="--",linewidth=2);
plt.text(kneePoint,-25,str(kneePoint));
plt.text(decade_kneePoint,-25,str(decade_kneePoint))
plt.axvline(decade_kneePoint, color="black", linestyle="--",linewidth=2)
plt.xlabel('bin separation wrt 2nd dataset');
plt.ylabel('standard deviation (dB)');
plt.grid(True);
plt.legend(snr_legend);


plt.figure(8,figsize=(20,10));
plt.title('Fractional error 99 percentile');
plt.plot(bin_sep_ls_samples,percentile_error_abs_tx2_dB[0,:,:].T,'-o');
plt.xscale('log')
plt.xlabel('bin separation wrt 2nd dataset');
plt.ylabel('99 percentile error (dB)');
plt.grid(True);
plt.legend(snr_legend);
# plt.xticks(bin_sep_ls_samples)


plt.figure(9,figsize=(20,10));
plt.title('Amplitude RMSE of 1st object from 10 chirps using LS');
plt.plot(bin_sep_ls_samples,rmse_error_amp_tx2[0,:,:].T,'-o');
plt.xscale('log')
plt.xlabel('bin separation wrt 2nd dataset');
plt.ylabel('RMSE (dB)');
plt.grid(True);
plt.legend(snr_legend);
# plt.xticks(bin_sep_ls_samples)



plt.figure(10,figsize=(20,10));
plt.title('Phase RMSE of 1st object from 10 chirps using LS');
plt.plot(bin_sep_ls_samples,rmse_error_angle_tx2[0,:,:].T,'-o');
plt.xscale('log')
plt.axvline(kneePoint, color="black", linestyle="--",linewidth=2);
plt.text(kneePoint,0.1,str(kneePoint));
plt.text(decade_kneePoint,0.1,str(decade_kneePoint))
plt.axvline(decade_kneePoint, color="black", linestyle="--",linewidth=2)
plt.xlabel('bin separation wrt 2nd dataset');
plt.ylabel('RMSE (rad)');
plt.grid(True);
plt.legend(snr_legend);



plt.figure(11,figsize=(20,10));
plt.title('Phase RMSE of 1st object from 10 chirps using LS');
plt.plot(bin_sep_ls_samples,rmse_error_angle_tx2_dB[0,:,:].T,'-o');
plt.xscale('log')
plt.axvline(kneePoint, color="black", linestyle="--",linewidth=2);
plt.text(kneePoint,-20,str(kneePoint));
plt.text(decade_kneePoint,-20,str(decade_kneePoint))
plt.axvline(decade_kneePoint, color="black", linestyle="--",linewidth=2)
plt.xlabel('bin separation wrt 2nd dataset');
plt.ylabel('RMSE (dB)');
plt.grid(True);
plt.legend(snr_legend);
# plt.xticks(bin_sep_ls_samples)
    
    
    
    
    
    
    
    
    
    
    
