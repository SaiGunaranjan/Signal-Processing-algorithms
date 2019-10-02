# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 23:29:48 2019

@author: Sai Gunaranjan Pelluri
"""
import numpy as np
from sai_omp import OMP as omp
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

num_sources = 2
object_snr = np.array([40,40])#np.random.randint(low=20,high=80,size=num_sources)
noise_power_db = -40 # Noise Power in dB
noise_variance = 10**(noise_power_db/10)
noise_sigma = np.sqrt(noise_variance)
weights = noise_variance*10**(object_snr/10)
num_signal_samples = 128
num_cols_signal_dict = 1500#num_signal_samples
num_subsampled_measurements = 32#num_signal_samples//4
num_rows_random_sampl_mat = num_subsampled_measurements
num_cols_random_sampl_mat = num_signal_samples
omp_threshold = 2e-1
eps = 2e-3
dig_freq_resol = 2*np.pi/num_cols_signal_dict 
freq_grid = np.linspace(-np.pi,np.pi,num_cols_signal_dict) #np.arange(-np.pi,np.pi,dig_freq_resol)#create a uniform grid of digital frequencies spaced 'dig_freq_resol' apart
sig_freq_ind = np.random.randint(num_cols_signal_dict, size=num_sources)#np.array([393,398])#np.random.randint(num_cols_signal_dict, size=num_sources) # select randomly from these discrete digital frequencies based on the number of sources given
sig_freq = freq_grid[sig_freq_ind] # select randomly from these discrete digital frequencies based on the number of sources given
signal_gen_matrix = np.exp(1j*np.matmul(np.arange(num_signal_samples)[:,None], freq_grid[None,:])) # Fat vandermode matrix from whose column space the signal is generated

#signal_gen_matrix = np.eye(num_signal_samples)

col_sampling_vector = np.zeros((num_cols_signal_dict)) 
#sig_freq_ind = np.random.randint(num_cols_signal_dict, size=num_sources) # select randomly from these discrete digital frequencies based on the number of sources given
col_sampling_vector[sig_freq_ind] = weights # choose weights for the sinusoids(columns) in the fat vandermonde matrix to be sampled
dopp_signal = np.matmul(signal_gen_matrix,col_sampling_vector)
noise_signal = np.random.normal(0,noise_sigma/np.sqrt(2),num_signal_samples) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),num_signal_samples) # generate a complex white gaissian noise
dopp_signal_with_noise = dopp_signal + noise_signal # signal + noise
ind_ones = np.sort(np.random.choice(range(num_cols_random_sampl_mat),num_rows_random_sampl_mat,replace=False))
random_sampling_matrix = np.zeros((num_rows_random_sampl_mat,num_cols_random_sampl_mat)).astype('complex64')
random_sampling_matrix[np.arange(num_rows_random_sampl_mat),ind_ones] = 1

#random_sampling_matrix = np.random.randn(num_rows_random_sampl_mat, num_cols_random_sampl_mat) + + 1j*np.random.randn(num_rows_random_sampl_mat, num_cols_random_sampl_mat) # create the random projection matrix which is a fat matrix with full row rank. In this case we have chosen iid gaussian matrix but we could choose bernoulli matrix as well

sub_sampl_signal = np.matmul(random_sampling_matrix,dopp_signal_with_noise)[:,None]
sub_sampl_signal = sub_sampl_signal/np.linalg.norm(sub_sampl_signal)
overall_mat = np.matmul(random_sampling_matrix,signal_gen_matrix)
sparse_coeff_vec, error_iter = omp(overall_mat, sub_sampl_signal, omp_threshold)
#sparse_coeff_vec, error_iter = omp(signal_gen_matrix, dopp_signal[:,None], omp_threshold)
sparse_coeff_vec = sparse_coeff_vec.squeeze()
sparse_coeff_vec[np.abs(sparse_coeff_vec)!=0] = 1
est_dopp_signal = np.matmul(signal_gen_matrix,sparse_coeff_vec)
sparse_coeff_vec_ls = (np.matmul(np.linalg.pinv(overall_mat),sub_sampl_signal)).squeeze()
#est_dopp_signal_ls = np.matmul(signal_gen_matrix,sparse_coeff_vec_ls)
local_max_ind_sparse_coeff_vec_ls = argrelextrema(np.abs(sparse_coeff_vec_ls), np.greater)[0]
sort_ind_local_max_ls = np.argsort(np.abs(sparse_coeff_vec_ls[local_max_ind_sparse_coeff_vec_ls]))
sort_ind_ls = local_max_ind_sparse_coeff_vec_ls[sort_ind_local_max_ls]
cols_est_ls = (sort_ind_ls[-num_sources::])
mod_sparse_coeff_vec_ls = np.zeros((num_cols_signal_dict)).astype('complex64') 
mod_sparse_coeff_vec_ls[cols_est_ls] = sparse_coeff_vec_ls[cols_est_ls]
est_dopp_signal_ls = np.matmul(signal_gen_matrix,mod_sparse_coeff_vec_ls)

fft_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/num_signal_samples)
plt.figure(10)
plt.plot(fft_freq_grid,20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(dopp_signal_with_noise)))),label='True Dopp Signal Spectrum')
plt.plot(fft_freq_grid,10+20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(est_dopp_signal)))),label='Est Dopp Signal Spectrum using OMP method')
plt.plot(fft_freq_grid,40+20*np.log10(eps+np.abs(np.fft.fftshift(np.fft.fft(est_dopp_signal_ls)))),label='Est Dopp Signal Spectrum using Least squares')
plt.vlines(sig_freq,-40,40, label = 'Ground truth')
plt.grid(True)
plt.legend()
#plt.figure(3)
#plt.imshow(np.abs(random_sampling_matrix),aspect='auto')


#plt.figure(4)
#plt.plot(np.abs(col_sampling_vector/np.amax(np.abs(col_sampling_vector))),label='True sparse vector')
#plt.plot(np.abs(sparse_coeff_vec),label='OMP reconstructed sparse vector')
#plt.plot(np.abs(sparse_coeff_vec_ls),'--.',label='Pseudo Inv reconstructed sparse vector')
#plt.grid(True)
#plt.legend()