# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 10:38:47 2022

@author: saiguna
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from digital_filter_functions import filterResponse2zeros, \
    impulseRespGen, generateVandermondeSuperSet, angular_distance

""" This script compares the performance of Least Squares, DFT, Zero modified transfer function (FIR)
in estimating the amplitude and phase of two sinusoids as the frequency separation between them varies from 0 to numFFT bins.
The results are analyzed for different SNRs"""



plt.close('all')

def compute_Twolargest_localpeaks(x):

    Ncases= x.shape[0]

    twoLargestLocalPeaks_matrix = np.zeros((0,2)).astype('int64')

    for i in range(Ncases):
        data=x[i,:]
        inx= argrelextrema(data,np.greater,order=2)[0]
        twoLargestLocalPeaks = inx[np.argsort(data[inx])][-2::]
        twoLargestLocalPeaks_matrix = np.vstack((twoLargestLocalPeaks,twoLargestLocalPeaks_matrix))

    return twoLargestLocalPeaks_matrix

def apes(received_signal, corr_mat_model_order, digital_freq_grid):
    '''corr_mat_model_order : must be strictly less than half then signal length'''
    signal_length = len(received_signal)
    auto_corr_matrix = np.zeros((corr_mat_model_order+1,corr_mat_model_order+1)).astype('complex64')
    y_tilda = np.zeros((corr_mat_model_order+1,0)).astype('complex64')
    for ele in np.arange(corr_mat_model_order,signal_length):
        if ele == corr_mat_model_order:
            auto_corr_matrix += np.matmul(received_signal[ele::-1,:],received_signal[ele::-1,:].T.conj())
            y_tilda = np.hstack((y_tilda,received_signal[ele::-1,:]))
        else:
            auto_corr_matrix += np.matmul(received_signal[ele:ele-corr_mat_model_order-1:-1,:],received_signal[ele:ele-corr_mat_model_order-1:-1,:].T.conj())
            y_tilda = np.hstack((y_tilda,received_signal[ele:ele-corr_mat_model_order-1:-1,:]))
    auto_corr_matrix = auto_corr_matrix/(signal_length-corr_mat_model_order) # Divide the auto-correlation matrix by the (signal length-corr_mat_model_order)
    auto_corr_matrix_inv = np.linalg.inv(auto_corr_matrix)
    vandermonde_matrix = np.exp(-1j*np.outer(np.arange(corr_mat_model_order+1),digital_freq_grid)) # [num_samples,num_freq] # construct the vandermond matrix for several uniformly spaced frequencies
    temp_phasor = np.exp(-1j*np.outer(np.arange(corr_mat_model_order, signal_length),digital_freq_grid))
    G_omega = np.matmul(y_tilda, temp_phasor)/(signal_length-corr_mat_model_order+1)
    Ah_Rinv_G = np.sum(vandermonde_matrix.conj()*np.matmul(auto_corr_matrix_inv, G_omega),axis=0)
    Gh_Rinv_G = np.sum(G_omega.conj()*np.matmul(auto_corr_matrix_inv, G_omega),axis=0)
    Ah_Rinv_A = np.sum(vandermonde_matrix.conj()*np.matmul(auto_corr_matrix_inv,vandermonde_matrix),axis=0)
    spectrum = Ah_Rinv_G/((1-Gh_Rinv_G)*Ah_Rinv_A + np.abs(Ah_Rinv_G)**2) # Actual APES based spectrum
#    spectrum = Ah_Rinv_G/Ah_Rinv_A # Capon based spectrum

    return spectrum



num_chirps_tx1 = 168;
num_chirps_tx2 = 14;
num_fft = 1024

numOrigSamp = num_fft
numSamp = num_chirps_tx2
vandermondeMatrixSuperSet = generateVandermondeSuperSet(numOrigSamp, numSamp)
zerosFilterResponseSuperSet = filterResponse2zeros(vandermondeMatrixSuperSet)


thermalNoise = -174 # dBm/Hz
noiseFigure = 10 # dB
baseBandgain = 34 #dB
adcSamplingRate = 35e6 # 35 MHz
dBFs_to_dBm = 10

noise_power_dbm = thermalNoise + noiseFigure + baseBandgain + 10*np.log10(adcSamplingRate); # Noise Power in dB
noise_power_dbFs = noise_power_dbm - 10 # dBm to dBFs
noiseFloorPerBin_dB = noise_power_dbFs - 10*np.log10(num_chirps_tx1)
noise_variance = 10**(noise_power_dbFs/10);
noise_sigma = np.sqrt(noise_variance);

bin_sep = np.arange(1,num_chirps_tx1-1,1)
bin_sep_ls_samples = (num_chirps_tx2/num_chirps_tx1)*bin_sep
SNR = np.linspace(10, 50, 9) # np.array([50]) # Ensure length of SNR array is a perfect square i.e. 2, 4, 9 so that we can put it as an N x N subplot
num_binsep = len(bin_sep)
num_snr = len(SNR)
num_montecarlo = 50#100#50#100;

num_objects = 2;
ampError_tx2 = np.zeros((num_montecarlo,num_objects,num_snr,num_binsep))
angleError_tx2 = np.zeros((num_montecarlo,num_objects,num_snr,num_binsep))
absError_tx2 = np.zeros((num_montecarlo,num_objects,num_snr,num_binsep))

ampError_tx2_dft = np.zeros((num_montecarlo,num_objects,num_snr,num_binsep))
angleError_tx2_dft = np.zeros((num_montecarlo,num_objects,num_snr,num_binsep))
absError_tx2_dft = np.zeros((num_montecarlo,num_objects,num_snr,num_binsep))

ampError_tx2_fir = np.zeros((num_montecarlo,num_objects,num_snr,num_binsep))
angleError_tx2_fir = np.zeros((num_montecarlo,num_objects,num_snr,num_binsep))
absError_tx2_fir = np.zeros((num_montecarlo,num_objects,num_snr,num_binsep))

freq_ind_1 = 0#np.random.randint(0,num_chirps_tx1-1);
cnt_bin = 0
for freq_ind_sep in bin_sep:
    freq_ind_perturb = np.random.uniform(0,0.5,1); #np.random.uniform(-0.5,0.5,1);
    freq_ind_1_perturb = freq_ind_1 + 0*freq_ind_perturb;
    freq_ind_2 = freq_ind_1_perturb + freq_ind_sep;
    freq_ind = np.array([freq_ind_1_perturb,freq_ind_2])
    freq_ind_tx2Scale = np.round((freq_ind/num_chirps_tx1)*num_fft).astype('int32')
    dig_freq = (2*np.pi*freq_ind)/num_chirps_tx1;


    cnt_snr = 0
    for snr in SNR:
        object_snr = np.array([snr,snr]);
        signal_amp_real = 10**((noiseFloorPerBin_dB + object_snr)/20)

        cnt_mc = 0
        for iter_num in np.arange(num_montecarlo):

            random_phase = np.hstack((0,2*np.pi*np.random.uniform(-0.5,0.5,1)))
            random_phasor = np.exp(1j*random_phase);
            true_complex_weights = signal_amp_real*random_phasor
            signal_tx1 = np.sum(np.exp(1j*np.matmul(dig_freq,np.arange(num_chirps_tx1)[None,:]))*true_complex_weights[:,None],axis=0);

            signal_tx2 = signal_tx1[0:num_chirps_tx2];

            # noise_tx1 = noise_sigma*np.sqrt(num_chirps_tx1)*(np.random.randn(num_chirps_tx1) + 1j*np.random.randn(num_chirps_tx1))
            noise_tx1 = noise_sigma*np.sqrt(2)*(np.random.randn(num_chirps_tx1) + 1j*np.random.randn(num_chirps_tx1))
            signal_noise_tx1 = signal_tx1 + noise_tx1;
            signal_noise_tx1_fft = np.fft.fft(np.hamming(num_chirps_tx1)*signal_noise_tx1,n=num_fft)/num_chirps_tx1;
            signal_noise_tx1_spectrum = 20*np.log10(np.abs(signal_noise_tx1_fft));
            est_freq_ind = compute_Twolargest_localpeaks(signal_noise_tx1_spectrum[None,:])[0]
            est_freq_ind = np.sort(est_freq_ind);
            est_dig_freq = (2*np.pi*est_freq_ind)/num_fft
            est_complex_weights_tx1_fft = signal_noise_tx1_fft[est_freq_ind];


            # noise_tx2 = noise_sigma*np.sqrt(2)*(np.random.randn(num_chirps_tx2) + 1j*np.random.randn(num_chirps_tx2))
            signal_noise_tx2 = signal_noise_tx1[0:num_chirps_tx2] # signal_tx2 + noise_tx2;


            signal_noise_tx2_fft = np.fft.fft(signal_noise_tx2,n=num_fft)/num_chirps_tx2;
            est_complex_weights_tx2_dft = (signal_noise_tx2_fft[freq_ind_tx2Scale.squeeze()]).squeeze()


            """ Below 2 lines of code is for APES based coefficient"""
            # corr_mat_model_order = num_chirps_tx2//2-2 # must be strictly less than num_samples/2
            # est_complex_weights_tx2_dft = apes(signal_noise_tx2[:,None], corr_mat_model_order, est_dig_freq)


            sigFreqInd_unipolar = freq_ind_tx2Scale[0,0] #est_freq_ind[0]
            interferFreqArray_bipolar = np.array([freq_ind_tx2Scale[1,0]])#np.array([est_freq_ind[1]])
            interferFreqArray_bipolar[interferFreqArray_bipolar>=num_fft//2] -= num_fft
            sig1_tf, zeros_sig1 = impulseRespGen(sigFreqInd_unipolar, interferFreqArray_bipolar, numOrigSamp, zerosFilterResponseSuperSet)
            signal1Coeff = np.matmul(sig1_tf, signal_noise_tx2)

            sigFreqInd_unipolar = freq_ind_tx2Scale[1,0]#est_freq_ind[1]
            interferFreqArray_bipolar = np.array([freq_ind_tx2Scale[0,0]])#np.array([est_freq_ind[0]])
            interferFreqArray_bipolar[interferFreqArray_bipolar>=num_fft//2] -= num_fft
            sig2_tf, zeros_sig2 = impulseRespGen(sigFreqInd_unipolar, interferFreqArray_bipolar, numOrigSamp, zerosFilterResponseSuperSet)
            signal2Coeff = np.matmul(sig2_tf, signal_noise_tx2)

            est_complex_weights_tx2_fir = np.array([signal1Coeff, signal2Coeff])


            """ Least Squares"""
            # vandermond_matrix = np.exp(1j*np.matmul(np.arange(num_chirps_tx2)[:,None],est_dig_freq[None,:]));
            vandermond_matrix = np.exp(1j*np.matmul(np.arange(num_chirps_tx2)[:,None],dig_freq.T));
            est_complex_weights_tx2 = np.matmul(np.linalg.pinv(vandermond_matrix),signal_noise_tx2);

            """ Error Statistics"""
            error_amp_tx2 = np.abs(true_complex_weights) - np.abs(est_complex_weights_tx2);
            # error_angle_tx2 = np.angle(true_complex_weights) - np.angle(est_complex_weights_tx2)
            error_angle_tx2 = angular_distance(np.angle(true_complex_weights), np.angle(est_complex_weights_tx2))
            error_abs_tx2 = np.abs((true_complex_weights - est_complex_weights_tx2)/true_complex_weights);

            error_amp_tx2_dft = np.abs(true_complex_weights) - np.abs(est_complex_weights_tx2_dft);
            # error_angle_tx2_dft = np.angle(true_complex_weights) - np.angle(est_complex_weights_tx2_dft)
            error_angle_tx2_dft = angular_distance(np.angle(true_complex_weights), np.angle(est_complex_weights_tx2_dft))
            error_abs_tx2_dft = np.abs((true_complex_weights - est_complex_weights_tx2_dft)/true_complex_weights);

            error_amp_tx2_fir = np.abs(true_complex_weights) - np.abs(est_complex_weights_tx2_fir);
            # error_angle_tx2_fir = np.angle(true_complex_weights) - np.angle(est_complex_weights_tx2_fir)
            error_angle_tx2_fir = angular_distance(np.angle(true_complex_weights), np.angle(est_complex_weights_tx2_fir))
            error_abs_tx2_fir = np.abs((true_complex_weights - est_complex_weights_tx2_fir)/true_complex_weights);


            ampError_tx2[cnt_mc,:,cnt_snr,cnt_bin] = error_amp_tx2;
            angleError_tx2[cnt_mc,:,cnt_snr,cnt_bin] = error_angle_tx2;
            absError_tx2[cnt_mc,:,cnt_snr,cnt_bin] = error_abs_tx2

            ampError_tx2_dft[cnt_mc,:,cnt_snr,cnt_bin] = error_amp_tx2_dft;
            angleError_tx2_dft[cnt_mc,:,cnt_snr,cnt_bin] = error_angle_tx2_dft;
            absError_tx2_dft[cnt_mc,:,cnt_snr,cnt_bin] = error_abs_tx2_dft

            ampError_tx2_fir[cnt_mc,:,cnt_snr,cnt_bin] = error_amp_tx2_fir;
            angleError_tx2_fir[cnt_mc,:,cnt_snr,cnt_bin] = error_angle_tx2_fir;
            absError_tx2_fir[cnt_mc,:,cnt_snr,cnt_bin] = error_abs_tx2_fir

            cnt_mc += 1;

        cnt_snr += 1;

    cnt_bin += 1



snr_legend = ['input SNR (dB) = ' + str(ele) for ele in SNR];

rmse_error_abs_tx2 = np.std(absError_tx2,axis=0)
percentile_error_abs_tx2 = np.percentile(absError_tx2,99,axis=0)
rmse_error_abs_tx2_dB = 20*np.log10(rmse_error_abs_tx2[0,:,:])
percentile_error_abs_tx2_dB = 20*np.log10(percentile_error_abs_tx2)

rmse_error_abs_tx2_dft = np.std(absError_tx2_dft,axis=0)
percentile_error_abs_tx2_dft = np.percentile(absError_tx2_dft,99,axis=0)
rmse_error_abs_tx2_dB_dft = 20*np.log10(rmse_error_abs_tx2_dft[0,:,:])
percentile_error_abs_tx2_dB_dft = 20*np.log10(percentile_error_abs_tx2_dft)

rmse_error_abs_tx2_fir = np.std(absError_tx2_fir,axis=0)
percentile_error_abs_tx2_fir = np.percentile(absError_tx2_fir,99,axis=0)
rmse_error_abs_tx2_dB_fir = 20*np.log10(rmse_error_abs_tx2_fir[0,:,:])
percentile_error_abs_tx2_dB_fir = 20*np.log10(percentile_error_abs_tx2_fir)




# rmse_error_amp_tx2 = 20*np.log10(np.sqrt(np.sum(ampError_tx2**2,axis=0)/(num_montecarlo)));
# rmse_error_angle_tx2 = np.sqrt(np.sum(angleError_tx2**2,axis=0)/(num_montecarlo));
# rmse_error_angle_tx2_dB = 20*np.log10(np.sqrt(np.sum(angleError_tx2**2,axis=0)/(num_montecarlo)));

# rmse_error_amp_tx2_dft = 20*np.log10(np.sqrt(np.sum(ampError_tx2_dft**2,axis=0)/(num_montecarlo)));
# rmse_error_angle_tx2_dft = np.sqrt(np.sum(angleError_tx2_dft**2,axis=0)/(num_montecarlo));
# rmse_error_angle_tx2_dB_dft = 20*np.log10(np.sqrt(np.sum(angleError_tx2_dft**2,axis=0)/(num_montecarlo)));

# rmse_error_amp_tx2_fir = 20*np.log10(np.sqrt(np.sum(ampError_tx2_fir**2,axis=0)/(num_montecarlo)));
# rmse_error_angle_tx2_fir = np.sqrt(np.sum(angleError_tx2_fir**2,axis=0)/(num_montecarlo));
# rmse_error_angle_tx2_dB_fir = 20*np.log10(np.sqrt(np.sum(angleError_tx2_fir**2,axis=0)/(num_montecarlo)));


rmse_error_amp_tx2 = 20*np.log10(np.std(ampError_tx2,axis=0));
rmse_error_angle_tx2 = np.std(angleError_tx2,axis=0);
rmse_error_angle_tx2_dB = 20*np.log10(rmse_error_angle_tx2);

rmse_error_amp_tx2_dft = 20*np.log10(np.std(ampError_tx2_dft,axis=0));
rmse_error_angle_tx2_dft = np.std(angleError_tx2_dft,axis=0);
rmse_error_angle_tx2_dB_dft = 20*np.log10(rmse_error_angle_tx2_dft);

rmse_error_amp_tx2_fir = 20*np.log10(np.std(ampError_tx2_fir,axis=0));
rmse_error_angle_tx2_fir = np.std(angleError_tx2_fir,axis=0);
rmse_error_angle_tx2_dB_fir = 20*np.log10(rmse_error_angle_tx2_fir);

kneePoint = 0.6
decade_kneePoint = kneePoint/10
xaxis_pts = np.array([decade_kneePoint,kneePoint])

if 0:
    plt.figure(7,figsize=(20,10));
    plt.title('Fractional error (dB)');
    plt.plot(bin_sep,rmse_error_abs_tx2_dB.T,'-o', label='LS');
    plt.plot(bin_sep,rmse_error_abs_tx2_dB_dft.T,'-o',label='DFT');
    # plt.xscale('log');
    # plt.axvline(kneePoint, color="black", linestyle="--",linewidth=2);
    # plt.text(kneePoint,-25,str(kneePoint));
    # plt.text(decade_kneePoint,-25,str(decade_kneePoint))
    # plt.axvline(decade_kneePoint, color="black", linestyle="--",linewidth=2)
    plt.xlabel('bin separation wrt 1st dataset');
    plt.ylabel('standard deviation (dB)');
    plt.grid(True);
    plt.legend();


    plt.figure(8,figsize=(20,10));
    plt.title('Fractional error 99 percentile');
    plt.plot(bin_sep,percentile_error_abs_tx2_dB[0,:,:].T,'-o', label='LS');
    plt.plot(bin_sep,percentile_error_abs_tx2_dB_dft[0,:,:].T,'-o', label='DFT');
    # plt.xscale('log')
    plt.xlabel('bin separation wrt 1st dataset');
    plt.ylabel('99 percentile error (dB)');
    plt.grid(True);
    plt.legend();
    # plt.xticks(bin_sep)


    plt.figure(9,figsize=(20,10));
    plt.subplot(1,2,1)
    plt.title('Amplitude RMSE of 1st object from ' + str(num_chirps_tx2) + ' chirps');
    plt.plot(bin_sep,rmse_error_amp_tx2[0,:,:].T,'-o',label='LS');
    plt.plot(bin_sep,rmse_error_amp_tx2_dft[0,:,:].T,'-o',label='DFT');
    # plt.xscale('log')
    plt.xlabel('bin separation wrt 1st dataset');
    plt.ylabel('RMSE (dB)');
    plt.grid(True);
    plt.legend();
    # plt.xticks(bin_sep)

    plt.subplot(1,2,2)
    plt.title('Amplitude RMSE of 2nd object from ' + str(num_chirps_tx2) + ' chirps');
    plt.plot(bin_sep,rmse_error_amp_tx2[1,:,:].T,'-o',label='LS');
    plt.plot(bin_sep,rmse_error_amp_tx2_dft[1,:,:].T,'-o',label='DFT');
    # plt.xscale('log')
    plt.xlabel('bin separation wrt 1st dataset');
    plt.ylabel('RMSE (dB)');
    plt.grid(True);
    plt.legend();
    # plt.xticks(bin_sep)



    plt.figure(10,figsize=(20,10));
    plt.subplot(1,2,1)
    plt.title('Phase RMSE of 1st object from ' + str(num_chirps_tx2) + ' chirps');
    plt.plot(bin_sep,rmse_error_angle_tx2[0,:,:].T,'-o',label='LS');
    plt.plot(bin_sep,rmse_error_angle_tx2_dft[0,:,:].T,'-o',label='DFT');
    plt.xlabel('bin separation wrt 1st dataset');
    plt.ylabel('RMSE (rad)');
    plt.grid(True);
    plt.legend();
    # plt.ylim([0,3])

    plt.subplot(1,2,2)
    plt.title('Phase RMSE of 2nd object from ' + str(num_chirps_tx2) + ' chirps');
    plt.plot(bin_sep,rmse_error_angle_tx2[1,:,:].T,'-o',label='LS');
    plt.plot(bin_sep,rmse_error_angle_tx2_dft[1,:,:].T,'-o',label='DFT');
    plt.xlabel('bin separation wrt 1st dataset');
    plt.ylabel('RMSE (rad)');
    plt.grid(True);
    plt.legend();
    # plt.ylim([0,3])



    plt.figure(11,figsize=(20,10));
    plt.subplot(1,2,1)
    plt.title('Phase RMSE(dB) of 1st object from ' + str(num_chirps_tx2) + ' chirps');
    plt.plot(bin_sep,rmse_error_angle_tx2_dB[0,:,:].T,'-o', label='LS');
    plt.plot(bin_sep,rmse_error_angle_tx2_dB_dft[0,:,:].T,'-o',label='DFT');
    plt.xlabel('bin separation wrt 1st dataset');
    plt.ylabel('RMSE (dB)');
    plt.grid(True);
    plt.legend();
    # plt.ylim([-25,10])

    plt.subplot(1,2,2)
    plt.title('Phase RMSE(dB) of 2nd object from ' + str(num_chirps_tx2) + ' chirps');
    plt.plot(bin_sep,rmse_error_angle_tx2_dB[1,:,:].T,'-o', label='LS');
    plt.plot(bin_sep,rmse_error_angle_tx2_dB_dft[1,:,:].T,'-o',label='DFT');
    plt.xlabel('bin separation wrt 1st dataset');
    plt.ylabel('RMSE (dB)');
    plt.grid(True);
    plt.legend();
    # plt.ylim([-25,10])


#####################################################################################

plt.figure(9,figsize=(20,10));
plt.suptitle('Amplitude RMSE of 1st object from ' + str(num_chirps_tx2) + ' chirps ' + 'vs bin separation wrt ' + str(num_chirps_tx1));
numRows = np.sqrt(len(SNR)).astype('int32')
numCols = numRows
for ele in range(len(SNR)):
    plt.subplot(numRows,numCols,ele+1)
    plt.title('SNR = ' + str(np.round(SNR[ele],2)) + ' dB');
    plt.plot(bin_sep,rmse_error_amp_tx2[0,ele,:],label='LS');
    plt.plot(bin_sep,rmse_error_amp_tx2_fir[0,ele,:],label='FIR');
    plt.plot(bin_sep,rmse_error_amp_tx2_dft[0,ele,:],label='DFT');
    # plt.xscale('log')
    # plt.xlabel('bin separation wrt 1st dataset');
    plt.ylabel('RMSE (dB)');
    plt.grid(True);
    plt.legend();
    # plt.ylim([-60, -40])


plt.figure(10,figsize=(20,10));
plt.suptitle('Amplitude RMSE of 2nd object from ' + str(num_chirps_tx2) + ' chirps ' + 'vs bin separation wrt ' + str(num_chirps_tx1));
numRows = np.sqrt(len(SNR)).astype('int32')
numCols = numRows
for ele in range(len(SNR)):
    plt.subplot(numRows,numCols,ele+1)
    plt.title('SNR = ' + str(np.round(SNR[ele],2)) + ' dB');
    plt.plot(bin_sep,rmse_error_amp_tx2[1,ele,:],label='LS');
    plt.plot(bin_sep,rmse_error_amp_tx2_fir[1,ele,:],label='FIR');
    plt.plot(bin_sep,rmse_error_amp_tx2_dft[1,ele,:],label='DFT');
    # plt.xscale('log')
    # plt.xlabel('bin separation wrt 1st dataset');
    plt.ylabel('RMSE (dB)');
    plt.grid(True);
    plt.legend();
    # plt.ylim([-60, -40])



plt.figure(11,figsize=(20,10));
plt.suptitle('Phase RMSE of 1st object from ' + str(num_chirps_tx2) + ' chirps ' + 'vs bin separation wrt ' + str(num_chirps_tx1));
numRows = np.sqrt(len(SNR)).astype('int32')
numCols = numRows
for ele in range(len(SNR)):
    plt.subplot(numRows,numCols,ele+1)
    plt.title('SNR = ' + str(np.round(SNR[ele],2)) + ' dB');
    plt.plot(bin_sep,rmse_error_angle_tx2[0,ele,:],label='LS');
    plt.plot(bin_sep,rmse_error_angle_tx2_fir[0,ele,:],label='FIR');
    plt.plot(bin_sep,rmse_error_angle_tx2_dft[0,ele,:],label='DFT');
    # plt.xscale('log')

    plt.ylabel('rad');
    plt.grid(True);
    plt.legend();


plt.figure(12,figsize=(20,10));
plt.suptitle('Phase RMSE of 2nd object from ' + str(num_chirps_tx2) + ' chirps ' + 'vs bin separation wrt ' + str(num_chirps_tx1));
numRows = np.sqrt(len(SNR)).astype('int32')
numCols = numRows
for ele in range(len(SNR)):
    plt.subplot(numRows,numCols,ele+1)
    plt.title('SNR = ' + str(np.round(SNR[ele],2)) + ' dB');
    plt.plot(bin_sep,rmse_error_angle_tx2[1,ele,:],label='LS');
    plt.plot(bin_sep,rmse_error_angle_tx2_fir[1,ele,:],label='FIR');
    plt.plot(bin_sep,rmse_error_angle_tx2_dft[1,ele,:],label='DFT');
    # plt.xscale('log')

    plt.ylabel('rad');
    plt.grid(True);
    plt.legend();



plt.figure(13,figsize=(20,10));
plt.suptitle('Phase RMSE(dB) of 1st object from ' + str(num_chirps_tx2) + ' chirps ' + 'vs bin separation wrt ' + str(num_chirps_tx1));
numRows = np.sqrt(len(SNR)).astype('int32')
numCols = numRows
for ele in range(len(SNR)):
    plt.subplot(numRows,numCols,ele+1)
    plt.title('SNR = ' + str(np.round(SNR[ele],2)) + ' dB');
    plt.plot(bin_sep,rmse_error_angle_tx2_dB[0,ele,:],label='LS');
    plt.plot(bin_sep,rmse_error_angle_tx2_dB_fir[0,ele,:],label='FIR');
    plt.plot(bin_sep,rmse_error_angle_tx2_dB_dft[0,ele,:],label='DFT');
    # plt.xscale('log')
    # plt.xlabel('bin separation wrt 1st dataset');
    plt.ylabel('RMSE (dB)');
    plt.grid(True);
    plt.legend();


plt.figure(14,figsize=(20,10));
plt.suptitle('Phase RMSE(dB) of 2nd object from ' + str(num_chirps_tx2) + ' chirps ' + 'vs bin separation wrt ' + str(num_chirps_tx1));
numRows = np.sqrt(len(SNR)).astype('int32')
numCols = numRows
for ele in range(len(SNR)):
    plt.subplot(numRows,numCols,ele+1)
    plt.title('SNR = ' + str(np.round(SNR[ele],2)) + ' dB');
    plt.plot(bin_sep,rmse_error_angle_tx2_dB[1,ele,:],label='LS');
    plt.plot(bin_sep,rmse_error_angle_tx2_dB_fir[1,ele,:],label='FIR');
    plt.plot(bin_sep,rmse_error_angle_tx2_dB_dft[1,ele,:],label='DFT');
    # plt.xscale('log')
    # plt.xlabel('bin separation wrt 1st dataset');
    plt.ylabel('RMSE (dB)');
    plt.grid(True);
    plt.legend();

