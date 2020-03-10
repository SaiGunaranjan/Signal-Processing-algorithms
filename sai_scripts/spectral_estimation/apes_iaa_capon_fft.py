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

def iaa_recursive(received_signal, digital_freq_grid, iterations):
    '''corr_mat_model_order : must be strictly less than half then signal length'''
    signal_length = len(received_signal)
    num_freq_grid_points = len(digital_freq_grid)    
    spectrum = np.fft.fftshift(np.fft.fft(received_signal.squeeze(),num_freq_grid_points)/(signal_length),axes=(0,))
#    spectrum = np.ones(num_freq_grid_points)
    vandermonde_matrix = np.exp(1j*np.outer(np.arange(signal_length),digital_freq_grid)) # [num_samples,num_freq] # construct the vandermond matrix for several uniformly spaced frequencies. Notice the posititve sign inside the exponential
    for iter_num in np.arange(iterations):
        spectrum_without_fftshift = np.fft.fftshift(spectrum,axes=(0,))
        power_vals = np.abs(spectrum_without_fftshift)**2
        double_sided_corr_vect = np.fft.fft(power_vals,num_freq_grid_points)/(num_freq_grid_points)
        single_sided_corr_vec = double_sided_corr_vect[0:signal_length] # r0,r1,..rM-1
        auto_corr_matrix = vtoeplitz(single_sided_corr_vec[None,:])[0,:,:].T
        auto_corr_matrix_inv = np.linalg.inv(auto_corr_matrix)
        Ah_Rinv_y = np.sum(vandermonde_matrix.conj()*np.matmul(auto_corr_matrix_inv, received_signal),axis=0)
        Ah_Rinv_A = np.sum(vandermonde_matrix.conj()*np.matmul(auto_corr_matrix_inv,vandermonde_matrix),axis=0)
        spectrum = Ah_Rinv_y/Ah_Rinv_A
        print(iter_num)
    return spectrum

   
#### Resolution analisis of Apes vs FFT 

plt.close('all')
num_samples = 32
num_sources = 2
object_snr = np.array([40,35])
noise_power_db = -40 # Noise Power in dB
noise_variance = 10**(noise_power_db/10)
noise_sigma = np.sqrt(noise_variance)
weights = noise_variance*(10**(object_snr/10))
signal_phases = np.exp(1j*np.random.uniform(low=-np.pi, high = np.pi, size = num_sources))
complex_signal_amplitudes = weights*signal_phases
random_freq = np.random.uniform(low=-np.pi, high=np.pi, size = 1)
fft_resol_fact = 2
resol_fact = 0.65
source_freq = np.array([random_freq, random_freq + resol_fact*2*np.pi/num_samples])
digital_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/(100*num_samples))
source_signals = np.matmul(np.exp(-1j*np.outer(np.arange(num_samples),source_freq)),complex_signal_amplitudes[:,None])
wgn_noise = np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape)
received_signal = source_signals + wgn_noise
#corr_mat_model_order = num_samples//2-2 # must be strictly less than num_samples/2
corr_mat_model_order = num_samples//2-2 # must be strictly less than num_samples/2

magnitude_spectrum_fft = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(received_signal,axis=0, n=len(digital_freq_grid))/received_signal.shape[0],axes=0)))
phase_spectrum_fft = np.unwrap(np.angle(np.fft.fftshift(np.fft.fft(received_signal,axis=0, n=len(digital_freq_grid))/received_signal.shape[0],axes=0)))

spectrum = apes(received_signal, corr_mat_model_order, digital_freq_grid)
magnitude_spectrum = np.abs(spectrum)
phase_spectrum = np.unwrap(np.angle(spectrum))
    
apes_est_freq_ind = np.argsort(magnitude_spectrum,axis=0)[-num_sources::]
estimated_complex_signal_amplitudes = spectrum[apes_est_freq_ind]


#    psd_f= capon_forward(received_signal, corr_mat_model_order, digital_freq_grid)
psd_b = capon_backward(received_signal, corr_mat_model_order, digital_freq_grid)
psd = capon_toeplitz(received_signal,digital_freq_grid);psd = np.flipud(psd);

iterations = 10
spectrum_iaa = iaa_recursive(received_signal, digital_freq_grid, iterations) # recursive IAA
magnitude_spectrum_iaa = np.abs(spectrum_iaa)    
    


print('\nTrue Signal Amplitudes:{}, APES based Signal Amplitudes: {}'.format(np.abs(complex_signal_amplitudes),np.abs(estimated_complex_signal_amplitudes)))
print('\nTrue Signal Phases(deg):{}, APES based Signal Phases(deg): {}'.format(np.angle(signal_phases)*180/np.pi,np.angle(estimated_complex_signal_amplitudes)*180/np.pi))

plt.figure(3)
plt.title('Magnitude Spectrum')
plt.plot(digital_freq_grid,10*np.log10(psd),alpha=0.7, label = 'CAPON toeplitz')
plt.plot(digital_freq_grid,10*np.log10(psd_b),alpha=0.7, label = 'CAPON Backward')
plt.plot(digital_freq_grid, 20*np.log10(magnitude_spectrum),alpha=0.7, label = 'APES');
plt.plot(digital_freq_grid, 20*np.log10(magnitude_spectrum_iaa),alpha=0.7, label = 'IAA')
plt.plot(digital_freq_grid, magnitude_spectrum_fft, label = 'FFT')
plt.vlines(-source_freq,-150,25, label = 'Ground truth')
plt.xlabel('Digital Frequencies')
plt.legend()    
plt.grid(True)







    


  




