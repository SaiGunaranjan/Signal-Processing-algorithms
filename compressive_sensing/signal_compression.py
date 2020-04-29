# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 23:34:33 2019

@author: Sai Gunaranjan Pelluri
"""
import numpy as np
import matplotlib.pyplot as plt
from compressive_sensing_lib import OMP as omp

""" This scheme has some logical bugs and results are not satisfactory. Need to debug this"""

plt.close('all')
num_signal_samples = 512
num_sources = 3
object_snr = np.random.randint(low=30,high=60,size=num_sources)#np.array([20,20,20,20,20])#np.random.randint(low=20,high=80,size=num_sources)
noise_power_db = -40 # Noise Power in dB
noiseFloorPerBin_dB = noise_power_db - 10*np.log10(num_signal_samples)
noise_variance = 10**(noise_power_db/10)
noise_sigma = np.sqrt(noise_variance)
# weights = noise_variance*10**(object_snr/10)
weights = np.sqrt(10**((noiseFloorPerBin_dB + object_snr)/10))
noise_VariancePerBin = 10**(noiseFloorPerBin_dB/10);
noiseSigmaPerBin = np.sqrt(noise_VariancePerBin)
num_cols_signal_dict = 1500

num_cols_random_proj_mat = num_signal_samples//2
num_rows_random_proj_mat = 30
omp_threshold = 10**((noiseFloorPerBin_dB+28)/10);
compr_fact = ((num_signal_samples-num_rows_random_proj_mat)/num_signal_samples)*100 


dig_freq_resol = np.pi/num_cols_signal_dict 
freq_grid = np.linspace(0,np.pi,num_cols_signal_dict) #create a uniform grid of digital frequencies spaced 'dig_freq_resol' apart
sig_freq_ind = np.random.randint(num_cols_signal_dict, size=num_sources) # select randomly from these discrete digital frequencies based on the number of sources given
sig_freq = freq_grid[sig_freq_ind] # select randomly from these discrete digital frequencies based on the number of sources given
signal_gen_matrix = np.exp(1j*np.matmul(np.arange(num_signal_samples)[:,None], freq_grid[None,:])) # Fat vandermode matrix from whose column space the signal is generated
col_sampling_vector = np.zeros((num_cols_signal_dict)) 
col_sampling_vector[sig_freq_ind] = weights # choose weights for the sinusoids(columns) in the fat vandermonde matrix to be sampled
range_signal = np.matmul(signal_gen_matrix,col_sampling_vector)*np.hanning(num_signal_samples) # generate the clean signal which is a sum of sinusoids with different weights coming from the column space of the fat vandermond/generating matrix
noise_signal = np.random.normal(0,noise_sigma/np.sqrt(2),num_signal_samples) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),num_signal_samples) # generate a complex white gaissian noise
range_signal_with_noise = range_signal + noise_signal # signal + noise
signal_spectrum = np.fft.fft(range_signal_with_noise)[0:num_signal_samples//2]/num_signal_samples # compute the fft of the noisy signal
random_projection_matrix = np.random.randn(num_rows_random_proj_mat, num_cols_random_proj_mat) + 1j*np.random.randn(num_rows_random_proj_mat, num_cols_random_proj_mat) # create the random projection matrix which is a fat matrix with full row rank. In this case we have chosen iid gaussian matrix but we could choose bernoulli matrix as well
random_projection_matrix = random_projection_matrix/np.linalg.norm(random_projection_matrix,axis=0)
random_proj_vec = np.matmul(random_projection_matrix, signal_spectrum)[:,None] # compute the random projections of the signal spectrum(which is sparse) on to the rows of the random projection matrix
recon_signal_spectrum, error_iter = omp(random_projection_matrix, random_proj_vec, omp_threshold) # solve for the sparse signal(signal spectrum in this case) using greedy algo like OMP using the random projection matrix and the random projections
num_zeros = recon_signal_spectrum.shape[0] - np.count_nonzero(recon_signal_spectrum,axis=0) # number of zero entries in the reconstructed signal (spectrum)
eps_noise = noiseSigmaPerBin*np.exp(1j*np.random.uniform(low=-np.pi,high=np.pi,size=num_zeros)) # add some small noise to the zero entries (just to compare the reconstructed signal spectrum with the true signal spectrum)
recon_signal_spectrum[np.abs(recon_signal_spectrum)==0] = eps_noise
plt.figure(1,figsize=(20,10))
plt.plot(np.arange(0,np.pi,2*np.pi/num_signal_samples), 20*np.log10(np.abs(signal_spectrum)),'-o',label='True Signal Spectrum')
plt.vlines(sig_freq,-100,0, label = 'Ground truth')
plt.plot(np.arange(0,np.pi,2*np.pi/num_signal_samples), 20*np.log10(np.abs(recon_signal_spectrum)),'-o',label='Reconstructed Signal Spectrum')
plt.grid(True)
plt.xlabel('Dig Freq (rad/samp)')
plt.legend()

print('\n\n Compression factor = ', compr_fact)

