# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 21:06:12 2019

@author: Sai Gunaranjan Pelluri
"""
import numpy as np
#import sai_spectral_estimation as spec_est
import matplotlib.pyplot as plt
import time

### Predefined functions

def sts_correlate(x):
    N= x.shape[1]
    y= np.hstack((x,np.zeros_like(x)))
    xfft=np.fft.fft(y,axis=1)
    corout= np.fft.ifft(xfft*np.conj(xfft),axis=1)[:,0:N]    
    return corout


def vtoeplitz(toprow):

    Npts= toprow.shape[1]
    Nrow= toprow.shape[0]
    
    ACM= np.zeros((Nrow,Npts,Npts)).astype('complex64')
    
    for i in range(Npts):
        ACM[:,i,i:]= toprow[:,0:Npts-i].conj()
        ACM[:,i:,i]= toprow[:,0:Npts-i]
    
    return ACM


def music_toeplitz(received_signal, num_sources, digital_freq_grid):
    signal_length = len(received_signal)
    auto_corr_vec = sts_correlate(received_signal.T) # Generate the auto-correlation vector of the same length as the signal
    auto_corr_matrix = vtoeplitz(auto_corr_vec) # Create a toeplitz matrix which is variant of the Auto-correlation matrix
    auto_corr_matrix = auto_corr_matrix[0,:,:]
    u, s, vh = np.linalg.svd(auto_corr_matrix) # Perform SVD of the Auto-correlation matrix
    noise_subspace = u[:,num_sources::] # The first # number of sources eigen vectors belong to the signal subspace and the remaining eigen vectors of U belong to the noise subspace which is orthogonal to the signal subspace. Hence pick these eigen vectors
    vandermonde_matrix = np.exp(-1j*np.outer(np.arange(signal_length),digital_freq_grid)) # [num_samples,num_freq] # construct the vandermond matrix for several uniformly spaced frequencies
    GhA = np.matmul(noise_subspace.T.conj(),vandermonde_matrix) #G*A essentially projects the vandermond matrix (which spans the signal subspace) on the noise subspace
    AhG = GhA.conj() # A*G
    AhGGhA = np.sum(AhG*GhA,axis=0) # A*GG*A
    pseudo_spectrum = 1/np.abs(AhGGhA) # Pseudo spectrum
    return pseudo_spectrum


def music_forward(received_signal, num_sources, corr_mat_model_order, digital_freq_grid):
    '''corr_mat_model_order : must be strictly less than half then signal length'''
    signal_length = len(received_signal)
    auto_corr_matrix = np.zeros((corr_mat_model_order,corr_mat_model_order)).astype('complex64')
    for ele in np.arange(num_samples-corr_mat_model_order+1): # model order dictates the length of the auto-correlation matrix
        auto_corr_matrix += np.matmul(received_signal[ele:ele+corr_mat_model_order,:],received_signal[ele:ele+corr_mat_model_order,:].T.conj()) # Generate the auto-correlation matrix using the expectation method. here we use the forward filtering i.e. y[0:m], y[1:m+1]...
    auto_corr_matrix = auto_corr_matrix/signal_length # Divide the auto-correlation matrix by the signal length
    u, s, vh = np.linalg.svd(auto_corr_matrix) # Perform SVD of the Auto-correlation matrix
    noise_subspace = u[:,num_sources::] # The first # number of sources eigen vectors belong to the signal subspace and the remaining eigen vectors of U belong to the noise subspace which is orthogonal to the signal subspace. Hence pick these eigen vectors
    vandermonde_matrix = np.exp(-1j*np.outer(np.arange(corr_mat_model_order),digital_freq_grid)) # [num_samples,num_freq] # construct the vandermond matrix for several uniformly spaced frequencies
    GhA = np.matmul(noise_subspace.T.conj(),vandermonde_matrix) #G*A essentially projects the vandermond matrix (which spans the signal subspace) on the noise subspace
    AhG = GhA.conj() # A*G
    AhGGhA = np.sum(AhG*GhA,axis=0) # A*GG*A
    pseudo_spectrum = 1/np.abs(AhGGhA) # Pseudo spectrum
    return pseudo_spectrum

def music_backward(received_signal, num_sources, corr_mat_model_order, digital_freq_grid):
    '''corr_mat_model_order : must be strictly less than half the signal length'''
    signal_length = len(received_signal)
    auto_corr_matrix = np.zeros((corr_mat_model_order,corr_mat_model_order)).astype('complex64')
    for ele in np.arange(corr_mat_model_order-1,signal_length):
        if ele == corr_mat_model_order-1:
            auto_corr_matrix += np.matmul(received_signal[ele::-1,:],received_signal[ele::-1,:].T.conj())
        else:
            auto_corr_matrix += np.matmul(received_signal[ele:ele-corr_mat_model_order:-1,:],received_signal[ele:ele-corr_mat_model_order:-1,:].T.conj())
    auto_corr_matrix = auto_corr_matrix/signal_length # Divide the auto-correlation matrix by the signal length
    u, s, vh = np.linalg.svd(auto_corr_matrix) # Perform SVD of the Auto-correlation matrix
    noise_subspace = u[:,num_sources::] # The first # number of sources eigen vectors belong to the signal subspace and the remaining eigen vectors of U belong to the noise subspace which is orthogonal to the signal subspace. Hence pick these eigen vectors
    vandermonde_matrix = np.exp(-1j*np.outer(np.arange(corr_mat_model_order),digital_freq_grid)) # [num_samples,num_freq] # construct the vandermond matrix for several uniformly spaced frequencies
    GhA = np.matmul(noise_subspace.T.conj(),vandermonde_matrix) #G*A essentially projects the vandermond matrix (which spans the signal subspace) on the noise subspace
    AhG = GhA.conj() # A*G
    AhGGhA = np.sum(AhG*GhA,axis=0) # A*GG*A
    pseudo_spectrum = 1/np.abs(AhGGhA) # Pseudo spectrum
    return pseudo_spectrum

def esprit_toeplitz(received_signal, num_sources):
    signal_length = len(received_signal)
    auto_corr_vec = sts_correlate(received_signal.T) # Generate the auto-correlation vector of the same length as the signal
    auto_corr_matrix = vtoeplitz(auto_corr_vec) # Create a toeplitz matrix which is variant of the Auto-correlation matrix
    auto_corr_matrix = auto_corr_matrix[0,:,:]
    u, s, vh = np.linalg.svd(auto_corr_matrix) # Perform SVD of the Auto-correlation matrix
    us = u[:,0:num_sources] # signal subspace
    us1 = us[0:signal_length-1,:] # First N-1 rows of us
    us2 = us[1:signal_length,:] # Last N-1 rows of us
    phi = np.matmul(np.linalg.pinv(us1), us2) # phi = pinv(us1)*us2, phi is similar to D and has same eigen vaues as D. D is a diagonal matrix with elements whose phase is the frequencies
    eig_vals = np.linalg.eigvals(phi) # compute eigen values of the phi matrix which are same as the eigen values of the D matrix since phi and D are similar matrices and hence share same eigen values
    est_freq = np.angle(eig_vals) # Angle/phase of the eigen values gives the frequencies
    return est_freq
    
def esprit_forward(received_signal, num_sources, corr_mat_model_order):
    '''corr_mat_model_order : must be strictly less than half then signal length'''
    signal_length = len(received_signal)
    auto_corr_matrix = np.zeros((corr_mat_model_order,corr_mat_model_order)).astype('complex64')
    for ele in np.arange(signal_length-corr_mat_model_order+1): # model order dictates the length of the auto-correlation matrix
        auto_corr_matrix += np.matmul(received_signal[ele:ele+corr_mat_model_order,:],received_signal[ele:ele+corr_mat_model_order,:].T.conj()) # Generate the auto-correlation matrix using the expectation method. here we use the forward filtering i.e. y[0:m], y[1:m+1]...
    auto_corr_matrix = auto_corr_matrix/signal_length # Divide the auto-correlation matrix by the signal length
    u, s, vh = np.linalg.svd(auto_corr_matrix) # Perform SVD of the Auto-correlation matrix
    us = u[:,0:num_sources] # signal subspace
    us1 = us[0:corr_mat_model_order-1,:] # First N-1 rows of us
    us2 = us[1:corr_mat_model_order,:] # Last N-1 rows of us
    phi = np.matmul(np.linalg.pinv(us1), us2) # phi = pinv(us1)*us2, phi is similar to D and has same eigen vaues as D. D is a diagonal matrix with elements whose phase is the frequencies
    eig_vals = np.linalg.eigvals(phi) # compute eigen values of the phi matrix which are same as the eigen values of the D matrix since phi and D are similar matrices and hence share same eigen values
    est_freq = np.angle(eig_vals) # Angle/phase of the eigen values gives the frequencies
    return est_freq   

def esprit_backward(received_signal, num_sources, corr_mat_model_order):
    '''corr_mat_model_order : must be strictly less than half then signal length'''
    signal_length = len(received_signal)
    auto_corr_matrix = np.zeros((corr_mat_model_order,corr_mat_model_order)).astype('complex64')
    for ele in np.arange(corr_mat_model_order-1,signal_length):
        if ele == corr_mat_model_order-1:
            auto_corr_matrix += np.matmul(received_signal[ele::-1,:],received_signal[ele::-1,:].T.conj())
        else:
            auto_corr_matrix += np.matmul(received_signal[ele:ele-corr_mat_model_order:-1,:],received_signal[ele:ele-corr_mat_model_order:-1,:].T.conj())
    auto_corr_matrix = auto_corr_matrix/signal_length # Divide the auto-correlation matrix by the signal length
    u, s, vh = np.linalg.svd(auto_corr_matrix) # Perform SVD of the Auto-correlation matrix
    us = u[:,0:num_sources] # signal subspace
    us1 = us[0:corr_mat_model_order-1,:] # First N-1 rows of us
    us2 = us[1:corr_mat_model_order,:] # Last N-1 rows of us
    phi = np.matmul(np.linalg.pinv(us1), us2) # phi = pinv(us1)*us2, phi is similar to D and has same eigen vaues as D. D is a diagonal matrix with elements whose phase is the frequencies
    eig_vals = np.linalg.eigvals(phi) # compute eigen values of the phi matrix which are same as the eigen values of the D matrix since phi and D are similar matrices and hence share same eigen values
    est_freq = np.angle(eig_vals) # Angle/phase of the eigen values gives the frequencies
    return est_freq 


def capon_toeplitz(received_signal, digital_freq_grid):
    signal_length = len(received_signal)
    auto_corr_vec = sts_correlate(received_signal.T) # Generate the auto-correlation vector of the same length as the signal
    auto_corr_matrix = vtoeplitz(auto_corr_vec) # Create a toeplitz matrix which is variant of the Auto-correlation matrix
    auto_corr_matrix = auto_corr_matrix[0,:,:]
    auto_corr_matrix_inv = np.linalg.inv(auto_corr_matrix)
    vandermonde_matrix = np.exp(-1j*np.outer(np.arange(signal_length),digital_freq_grid)) # [num_samples,num_freq] # construct the vandermond matrix for several uniformly spaced frequencies
    auto_corr_matrix_inv_pow_2 = np.matmul(auto_corr_matrix_inv,auto_corr_matrix_inv)
#    filter_bw_beta = corr_mat_model_order + 1
    Ah_Rinv_2_A = np.sum(vandermonde_matrix.conj()*np.matmul(auto_corr_matrix_inv_pow_2,vandermonde_matrix),axis=0)
    Ah_Rinv_A = np.sum(vandermonde_matrix.conj()*np.matmul(auto_corr_matrix_inv,vandermonde_matrix),axis=0)
    filter_bw_beta = Ah_Rinv_2_A/(Ah_Rinv_A)**2
    psd = np.abs((1/(Ah_Rinv_A))/filter_bw_beta)
    return psd

def capon_forward(received_signal, corr_mat_model_order, digital_freq_grid):
    '''corr_mat_model_order : must be strictly less than half then signal length'''
    signal_length = len(received_signal)
    auto_corr_matrix = np.zeros((corr_mat_model_order,corr_mat_model_order)).astype('complex64')
    for ele in np.arange(signal_length-corr_mat_model_order+1): # model order dictates the length of the auto-correlation matrix
        auto_corr_matrix += np.matmul(received_signal[ele:ele+corr_mat_model_order,:],received_signal[ele:ele+corr_mat_model_order,:].T.conj()) # Generate the auto-correlation matrix using the expectation method. here we use the forward filtering i.e. y[0:m], y[1:m+1]...
    auto_corr_matrix = auto_corr_matrix/(signal_length-corr_mat_model_order) # Divide the auto-correlation matrix by the (signal length-corr_mat_model_order)
    auto_corr_matrix_inv = np.linalg.inv(auto_corr_matrix)
    vandermonde_matrix = np.exp(-1j*np.outer(np.arange(corr_mat_model_order),digital_freq_grid)) # [num_samples,num_freq] # construct the vandermond matrix for several uniformly spaced frequencies
    auto_corr_matrix_inv_pow_2 = np.matmul(auto_corr_matrix_inv,auto_corr_matrix_inv)
    Ah_Rinv_2_A = np.sum(vandermonde_matrix.conj()*np.matmul(auto_corr_matrix_inv_pow_2,vandermonde_matrix),axis=0)
    Ah_Rinv_A = np.sum(vandermonde_matrix.conj()*np.matmul(auto_corr_matrix_inv,vandermonde_matrix),axis=0)
#    filter_bw_beta = Ah_Rinv_2_A/(Ah_Rinv_A)**2
    filter_bw_beta = corr_mat_model_order + 1
    psd = np.abs((1/(Ah_Rinv_A))/filter_bw_beta)
    return psd
    
    
def capon_backward(received_signal, corr_mat_model_order, digital_freq_grid):
    '''corr_mat_model_order : must be strictly less than half then signal length'''
    signal_length = len(received_signal)
    auto_corr_matrix = np.zeros((corr_mat_model_order+1,corr_mat_model_order+1)).astype('complex64')
    for ele in np.arange(corr_mat_model_order,signal_length):
        if ele == corr_mat_model_order:
            auto_corr_matrix += np.matmul(received_signal[ele::-1,:],received_signal[ele::-1,:].T.conj())
        else:
            auto_corr_matrix += np.matmul(received_signal[ele:ele-corr_mat_model_order-1:-1,:],received_signal[ele:ele-corr_mat_model_order-1:-1,:].T.conj())
    auto_corr_matrix = auto_corr_matrix/(signal_length-corr_mat_model_order) # Divide the auto-correlation matrix by the (signal length-corr_mat_model_order)
    auto_corr_matrix_inv = np.linalg.inv(auto_corr_matrix)
    vandermonde_matrix = np.exp(-1j*np.outer(np.arange(corr_mat_model_order+1),digital_freq_grid)) # [num_samples,num_freq] # construct the vandermond matrix for several uniformly spaced frequencies
    auto_corr_matrix_inv_pow_2 = np.matmul(auto_corr_matrix_inv,auto_corr_matrix_inv)
    Ah_Rinv_2_A = np.sum(vandermonde_matrix.conj()*np.matmul(auto_corr_matrix_inv_pow_2,vandermonde_matrix),axis=0)
    Ah_Rinv_A = np.sum(vandermonde_matrix.conj()*np.matmul(auto_corr_matrix_inv,vandermonde_matrix),axis=0)
    filter_bw_beta = corr_mat_model_order + 1
#    filter_bw_beta = Ah_Rinv_2_A/(Ah_Rinv_A)**2
    psd = np.abs((1/(Ah_Rinv_A))/filter_bw_beta)
    return psd    



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










#######














plt.close('all')

SNR = np.arange(25,26)#np.arange(-5,25,2)
num_samples = 256#1024
signal_power = 1
num_montecarlo = 3#100
num_sources = 2
corr_mat_model_order = 100 # corr_mat_model_order : must be strictly less than half the signal length
fft_freq_resol = 2*np.pi/num_samples
min_dig_freq_step_size = fft_freq_resol/10
digital_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/(100*num_samples))
dig_freq_steps = np.arange(min_dig_freq_step_size,fft_freq_resol,min_dig_freq_step_size)

rmse_fft_snr = np.zeros((0,len(dig_freq_steps)))
rmse_music_snr = np.zeros((0,len(dig_freq_steps)))
rmse_esprit_snr = np.zeros((0,len(dig_freq_steps)))
rmse_capon_snr = np.zeros((0,len(dig_freq_steps)))
rmse_apes_snr = np.zeros((0,len(dig_freq_steps)))

rmse_fft_snr_sec_source = np.zeros((0,len(dig_freq_steps)))
rmse_music_snr_sec_source = np.zeros((0,len(dig_freq_steps)))
rmse_esprit_snr_sec_source = np.zeros((0,len(dig_freq_steps)))
rmse_capon_snr_sec_source = np.zeros((0,len(dig_freq_steps)))
rmse_apes_snr_sec_source = np.zeros((0,len(dig_freq_steps)))
tstart = time.time()
for snr in SNR:
    noise_power = 10**(-snr/10)
    noise_sigma = np.sqrt(noise_power)
    error_fft = []
    error_music = []
    error_esprit = []
    error_capon = []
    error_apes = []
    error_fft_sec_source = []
    error_music_sec_source = []
    error_esprit_sec_source = []
    error_capon_sec_source = []
    error_apes_sec_source = []
    
    rmse_fft_res = []
    rmse_music_res = []
    rmse_esprit_res = []
    rmse_capon_res = []
    rmse_apes_res = []
    rmse_fft_res_sec_source = []
    rmse_music_res_sec_source = []
    rmse_esprit_res_sec_source = []
    rmse_capon_res_sec_source = []
    rmse_apes_res_sec_source = []
    
    for dig_freq_delta in dig_freq_steps:
        dig_freq = np.random.uniform(-np.pi, np.pi)
        second_source_dig_freq = dig_freq + dig_freq_delta
        signal1 = np.exp(1j*dig_freq*np.arange(num_samples))
        signal2 = np.exp(1j*second_source_dig_freq*np.arange(num_samples))
        signal = (signal1 + signal2)/np.sqrt(num_samples)
        for iteration in np.arange(num_montecarlo):
            noise = np.random.normal(0,noise_sigma/np.sqrt(2),num_samples) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),num_samples)
            noisy_signal = (signal + noise)[:,None]
            fft_indices = np.argsort(np.abs(np.fft.fft(noisy_signal,axis=0)),axis=0)[-num_sources::]
            est_dig_freq_fft = (2*np.pi/num_samples)*np.array([fft_index if fft_index <= num_samples//2 else fft_index-num_samples for fft_index in fft_indices])
            pseudo_spectrum = music_backward(noisy_signal, num_sources, corr_mat_model_order, digital_freq_grid)
            est_dig_freq_music = digital_freq_grid[np.argsort(pseudo_spectrum)[-num_sources::]]
            est_dig_freq_esprit = -1*esprit_backward(noisy_signal, num_sources, corr_mat_model_order)
            psd_b = capon_backward(noisy_signal, corr_mat_model_order, digital_freq_grid)
            est_dig_freq_capon = digital_freq_grid[np.argsort(psd_b)[-num_sources::]]
            spectrum = apes(noisy_signal, corr_mat_model_order, digital_freq_grid)
            magnitude_spectrum = np.abs(spectrum)
            est_dig_freq_apes = digital_freq_grid[np.argsort(magnitude_spectrum)[-num_sources::]]
            
            error_fft.append(dig_freq-np.amin(est_dig_freq_fft))
            error_music.append(dig_freq-np.amin(est_dig_freq_music))
            error_esprit.append(dig_freq-np.amin(est_dig_freq_esprit))
            error_capon.append(dig_freq-np.amin(est_dig_freq_capon))
            error_apes.append(dig_freq-np.amin(est_dig_freq_apes))
    
            error_fft_sec_source.append(second_source_dig_freq-np.amax(est_dig_freq_fft))
            error_music_sec_source.append(second_source_dig_freq-np.amax(est_dig_freq_music))
            error_esprit_sec_source.append(second_source_dig_freq-np.amax(est_dig_freq_esprit))
            error_capon_sec_source.append(second_source_dig_freq-np.amax(est_dig_freq_capon))
            error_apes_sec_source.append(second_source_dig_freq-np.amax(est_dig_freq_apes))
        
        rmse_fft =  np.sqrt(np.sum(np.array(error_fft)**2)/num_montecarlo)
        rmse_music =  np.sqrt(np.sum(np.array(error_music)**2)/num_montecarlo)
        rmse_esprit =  np.sqrt(np.sum(np.array(error_esprit)**2)/num_montecarlo)
        rmse_capon =  np.sqrt(np.sum(np.array(error_capon)**2)/num_montecarlo)
        rmse_apes =  np.sqrt(np.sum(np.array(error_apes)**2)/num_montecarlo)
    
        rmse_fft_sec_source =  np.sqrt(np.sum(np.array(error_fft_sec_source)**2)/num_montecarlo)
        rmse_music_sec_source =  np.sqrt(np.sum(np.array(error_music_sec_source)**2)/num_montecarlo)
        rmse_esprit_sec_source =  np.sqrt(np.sum(np.array(error_esprit_sec_source)**2)/num_montecarlo)
        rmse_capon_sec_source =  np.sqrt(np.sum(np.array(error_capon_sec_source)**2)/num_montecarlo)
        rmse_apes_sec_source =  np.sqrt(np.sum(np.array(error_apes_sec_source)**2)/num_montecarlo)
        
        rmse_fft_res.append(rmse_fft)
        rmse_music_res.append(rmse_music)
        rmse_esprit_res.append(rmse_esprit)
        rmse_capon_res.append(rmse_capon)
        rmse_apes_res.append(rmse_apes)
    
    
        rmse_fft_res_sec_source.append(rmse_fft_sec_source)
        rmse_music_res_sec_source.append(rmse_music_sec_source)
        rmse_esprit_res_sec_source.append(rmse_esprit_sec_source)
        rmse_capon_res_sec_source.append(rmse_capon_sec_source)
        rmse_apes_res_sec_source.append(rmse_apes_sec_source)
        
        
    
    rmse_fft_snr = np.vstack((rmse_fft_snr,rmse_fft_res))
    rmse_music_snr = np.vstack((rmse_music_snr,rmse_music_res))
    rmse_esprit_snr = np.vstack((rmse_esprit_snr,rmse_esprit_res))
    rmse_capon_snr = np.vstack((rmse_capon_snr,rmse_capon_res))
    rmse_apes_snr = np.vstack((rmse_apes_snr,rmse_apes_res))


    rmse_fft_snr_sec_source = np.vstack((rmse_fft_snr_sec_source,rmse_fft_res_sec_source))
    rmse_music_snr_sec_source = np.vstack((rmse_music_snr_sec_source,rmse_music_res_sec_source))
    rmse_esprit_snr_sec_source = np.vstack((rmse_esprit_snr_sec_source,rmse_esprit_res_sec_source))
    rmse_capon_snr_sec_source = np.vstack((rmse_capon_snr_sec_source,rmse_capon_res_sec_source))
    rmse_apes_snr_sec_source = np.vstack((rmse_apes_snr_sec_source,rmse_apes_res_sec_source))
    
tend = time.time()
print("Time elapsed = {} secs".format(np.round((tend-tstart)*10)/10))


      
plt.figure(1)
plt.subplot(121)
plt.title('First source')
#plt.plot(dig_freq_steps, rmse_fft_snr.squeeze(),'o-',label='FFT')
#plt.plot(dig_freq_steps, rmse_music_snr.squeeze(),'o-',label='MUSIC')
#plt.plot(dig_freq_steps, rmse_esprit_snr.squeeze(),'o-',label='ESPRIT')
plt.plot(dig_freq_steps, rmse_capon_snr.squeeze(),'o-',label='CAPON')
plt.plot(dig_freq_steps, rmse_apes_snr.squeeze(),'o-',label='APES')
plt.xlabel('Digital Freq Resol (rad/samp)')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)

plt.subplot(122)
plt.title('Second source')
#plt.plot(dig_freq_steps, rmse_fft_snr_sec_source.squeeze(),'o-',label='FFT')
#plt.plot(dig_freq_steps, rmse_music_snr_sec_source.squeeze(),'o-',label='MUSIC')
#plt.plot(dig_freq_steps, rmse_esprit_snr_sec_source.squeeze(),'o-',label='ESPRIT')
plt.plot(dig_freq_steps, rmse_capon_snr_sec_source.squeeze(),'o-',label='CAPON')
plt.plot(dig_freq_steps, rmse_apes_snr_sec_source.squeeze(),'o-',label='APES')
plt.xlabel('Digital Freq Resol (rad/samp)')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
