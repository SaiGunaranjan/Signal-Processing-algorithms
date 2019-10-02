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


def solve_levinson_durbin(toeplitz_matrix, y_vec):
    '''
    Solves for Tx = y
     inputs:
         toeplitz matrix (T): NxN
         y_vec : numpy array of length N
     outputs:
         solution vector x: numpy array of length N
     
        Refer wiki page: https://en.wikipedia.org/wiki/Levinson_recursion # for a simple and elegant understanding and implemenation of the algo
        Refer https://github.com/topisani/OTTO/pull/104/files/c5985545bb39de2a27689066150a5caac0c1fdf9 for the cpp implemenation of the algo
    
    '''
    corr_mat = toeplitz_matrix.copy()
    num_iter = corr_mat.shape[0] # N
    inv_fact = 1/corr_mat[0,0] # 1/t0
    backward_vec = inv_fact
    forward_vec = inv_fact
    x_vec = y_vec[0]*inv_fact # x_vec = y[0]/t0
    for iter_count in np.arange(2,num_iter+1):
        forward_error = np.dot(corr_mat[iter_count-1:0:-1,0],forward_vec) # inner product between the forward vector from previous iteration and a flipped version of the 0th column of the corr_mat 
        backward_error = np.dot(corr_mat[0,1:iter_count],backward_vec) # inner product between the backward vector from previous iteration and the 0th row of the corr_mat 
        error_fact = 1/(1-(backward_error*forward_error))
        prev_iter_forward_vec = forward_vec.copy()
        forward_vec = error_fact*np.append(forward_vec,0) - forward_error*error_fact*np.append(0,backward_vec) # forward vector update
        backward_vec = error_fact*np.append(0,backward_vec) - backward_error*error_fact*np.append(prev_iter_forward_vec,0) # backward vector update
        error_x_vec = np.dot(x_vec,corr_mat[iter_count-1,0:iter_count-1]) # error in the xvector
        x_vec = np.append(x_vec,0) + (y_vec[iter_count-1]-error_x_vec)*backward_vec # x_vec update
        
    return x_vec

def solve_levinson_durbin_ymatrix(toeplitz_matrix, y_vector):
    '''
    Solves for Tx = y
     inputs:
         toeplitz matrix (T): NxN
         y_vec : numpy array of shape N x M : M is the number of different y's
     outputs:
         solution vector x: numpy array of length N
     
        Refer wiki page: https://en.wikipedia.org/wiki/Levinson_recursion # for a simple and elegant understanding and implemenation of the algo
        Refer https://github.com/topisani/OTTO/pull/104/files/c5985545bb39de2a27689066150a5caac0c1fdf9 for the cpp implemenation of the algo
    
    '''
    corr_mat = toeplitz_matrix.copy()
    y_vec = y_vector.copy()
    num_iter = corr_mat.shape[0] # N
    inv_fact = 1/corr_mat[0,0] # 1/t0
    num_sols = y_vec.shape[1]
    x_mat = np.zeros((0,num_iter)).astype('complex64')
    for ele in np.arange(num_sols):
        backward_vec = inv_fact
        forward_vec = inv_fact
        x_vec = y_vec[0,ele]*inv_fact # x_vec = y[0]/t0
        for iter_count in np.arange(2,num_iter+1):
            forward_error = np.dot(corr_mat[iter_count-1:0:-1,0],forward_vec) # inner product between the forward vector from previous iteration and a flipped version of the 0th column of the corr_mat 
            backward_error = np.dot(corr_mat[0,1:iter_count],backward_vec) # inner product between the backward vector from previous iteration and the 0th row of the corr_mat 
            error_fact = 1/(1-(backward_error*forward_error))
            prev_iter_forward_vec = forward_vec.copy()
            forward_vec = error_fact*np.append(forward_vec,0) - forward_error*error_fact*np.append(0,backward_vec) # forward vector update
            backward_vec = error_fact*np.append(0,backward_vec) - backward_error*error_fact*np.append(prev_iter_forward_vec,0) # backward vector update
            error_x_vec = np.dot(x_vec,corr_mat[iter_count-1,0:iter_count-1]) # error in the xvector
            x_vec = np.append(x_vec,0) + (y_vec[iter_count-1,ele]-error_x_vec)*backward_vec # x_vec update
        x_mat = np.vstack((x_mat,x_vec))
    final_x_mat = x_mat.T
    
    return final_x_mat


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


def iaa_approx_nonrecursive(received_signal, digital_freq_grid):
    '''corr_mat_model_order : must be strictly less than half then signal length'''
    signal_length = len(received_signal)
    auto_corr_vec = sts_correlate(received_signal.T) # Generate the auto-correlation vector of the same length as the signal
    auto_corr_matrix = vtoeplitz(auto_corr_vec) # Create a toeplitz matrix which is variant of the Auto-correlation matrix
    auto_corr_matrix = auto_corr_matrix[0,:,:]    
    auto_corr_matrix_inv = np.linalg.inv(auto_corr_matrix)
    vandermonde_matrix = np.exp(1j*np.outer(np.arange(signal_length),digital_freq_grid)) # [num_samples,num_freq] # construct the vandermond matrix for several uniformly spaced frequencies. Notice the posititve sign inside the exponential
    Ah_Rinv_y = np.sum(vandermonde_matrix.conj()*np.matmul(auto_corr_matrix_inv, received_signal),axis=0)
    Ah_Rinv_A = np.sum(vandermonde_matrix.conj()*np.matmul(auto_corr_matrix_inv,vandermonde_matrix),axis=0)
#    spectrum = Ah_Rinv_G/((1-Gh_Rinv_G)*Ah_Rinv_A + np.abs(Ah_Rinv_G)**2) # Actual APES based spectrum
    spectrum = Ah_Rinv_y/Ah_Rinv_A 
    
    return spectrum


def iaa_approx_recursive_computeheavy(received_signal, digital_freq_grid, iterations):
    '''corr_mat_model_order : must be strictly less than half then signal length'''
    signal_length = len(received_signal)
    num_freq_grid_points = len(digital_freq_grid)
    spectrum = np.fft.fftshift(np.fft.fft(received_signal.squeeze(),num_freq_grid_points)/(signal_length),axes=(0,))
    vandermonde_matrix = np.exp(1j*np.outer(np.arange(signal_length),digital_freq_grid)) # [num_samples,num_freq] # construct the vandermond matrix for several uniformly spaced frequencies. Notice the posititve sign inside the exponential
    for iter_num in np.arange(iterations):
        power_vals = np.abs(spectrum)**2
        diagonal_mat_power_vals = np.diag(power_vals)
        A_P_Ah = np.matmul(vandermonde_matrix,np.matmul(diagonal_mat_power_vals,vandermonde_matrix.T.conj()))
        auto_corr_matrix = A_P_Ah.copy()
        auto_corr_matrix_inv = np.linalg.inv(auto_corr_matrix)
        Ah_Rinv_y = np.sum(vandermonde_matrix.conj()*np.matmul(auto_corr_matrix_inv, received_signal),axis=0)
        Ah_Rinv_A = np.sum(vandermonde_matrix.conj()*np.matmul(auto_corr_matrix_inv,vandermonde_matrix),axis=0)
        spectrum = Ah_Rinv_y/Ah_Rinv_A 
    
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


def iaa_recursive_levinson_temp(received_signal, digital_freq_grid, iterations):
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
#        auto_corr_matrix_inv = np.linalg.inv(auto_corr_matrix)
        Rinv_y = solve_levinson_durbin(auto_corr_matrix, received_signal.squeeze())
        Ah_Rinv_y = np.sum(vandermonde_matrix.conj()*Rinv_y[:,None],axis=0)
        Rinv_A = solve_levinson_durbin_ymatrix(auto_corr_matrix, vandermonde_matrix)
        Ah_Rinv_A = np.sum(vandermonde_matrix.conj()*Rinv_A,axis=0)
        spectrum = Ah_Rinv_y/Ah_Rinv_A
        print(iter_num)
    return spectrum



if 0:
    plt.close('all')
    num_samples = 1024
    num_sources = 3
    object_snr = np.array([20,20,20])
    noise_power_db = -40 # Noise Power in dB
    noise_variance = 10**(noise_power_db/10)
    noise_sigma = np.sqrt(noise_variance)
    weights = noise_variance*10**(object_snr/10)
    source_freq = np.random.uniform(low=-np.pi, high=np.pi, size = num_sources)
    source_signals = np.matmul(np.exp(-1j*np.outer(np.arange(num_samples),source_freq)),weights[:,None])
    wgn_noise = np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape)
    received_signal = source_signals + wgn_noise
    corr_mat_model_order = 100 # must be strictly less than num_samples/2
    digital_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/num_samples)

if 0:
    #pseudo_spectrum = music_toeplitz(received_signal, num_sources, digital_freq_grid)
#    pseudo_spectrum = music_forward(received_signal, num_sources, corr_mat_model_order, digital_freq_grid)
    pseudo_spectrum = music_backward(received_signal, num_sources, corr_mat_model_order, digital_freq_grid)
    print('\n\n')
    print('True Frequencies:', source_freq, 'Estimated Frequncies:', digital_freq_grid[np.argsort(pseudo_spectrum)[-num_sources::]])
    plt.figure(1)
    plt.title('Pseudo Spectrum from MUSIC')
    plt.plot(digital_freq_grid,10*np.log10(pseudo_spectrum),'o-',alpha=0.7)
    plt.vlines(-source_freq,-30,20)
    plt.xlabel('Digital Frequencies')
    plt.legend(['Estimated Pseudo Spectrum', 'Ground Truth'])
    plt.grid(True)

if 0: 
    #est_freq = esprit_toeplitz(received_signal, num_sources)
#    est_freq = esprit_forward(received_signal, num_sources, corr_mat_model_order)
    est_freq = esprit_backward(received_signal, num_sources, corr_mat_model_order)
    print('True Frequencies:', source_freq, 'Estimated Frequncies:', -est_freq)
if 0:
#    psd_f, digital_freq_grid = capon_forward(received_signal, corr_mat_model_order, digital_freq_grid)
    psd_b = capon_backward(received_signal, corr_mat_model_order, digital_freq_grid)
    #psd, digital_freq_grid = capon_toeplitz(received_signal)
    plt.figure(2)
    plt.title('Power spectral density from Capon')
#    plt.plot(digital_freq_grid,10*np.log10(psd_f),'o-',alpha=0.7, label = 'CAPON Forward')
    plt.plot(digital_freq_grid,10*np.log10(psd_b),'o-',alpha=0.7, label = 'CAPON Backward')
#    plt.vlines(-source_freq,-80,-70)
    plt.plot(digital_freq_grid, 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(received_signal,axis=0),axes=0))),'o-', label = 'FFT')
    plt.legend()
    plt.xlabel('Digital Frequencies')
    
    plt.grid(True)
    
    
#### Resolution analisis of Apes vs FFT vs approximate non-recursive IAA vs recursive IAA   
if 1:
    plt.close('all')
    num_samples = 32
    num_sources = 2
    object_snr = np.array([40,40])
    noise_power_db = -40 # Noise Power in dB
    noise_variance = 10**(noise_power_db/10)
    noise_sigma = np.sqrt(noise_variance)
    weights = noise_variance*(10**(object_snr/10))
    signal_phases = np.exp(1j*np.random.uniform(low=-np.pi, high = np.pi, size = num_sources))
    complex_signal_amplitudes = weights*signal_phases
    random_freq = np.random.uniform(low=-np.pi, high=np.pi, size = 1)
    fft_resol_fact = 2
    resol_fact = 0.53#0.65
    source_freq = np.array([random_freq, random_freq + resol_fact*np.pi/num_samples])
    digital_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/(100*num_samples))
    source_signals = np.matmul(np.exp(-1j*np.outer(np.arange(num_samples),source_freq)),complex_signal_amplitudes[:,None])
    wgn_noise = np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape)
    received_signal = source_signals + wgn_noise
    corr_mat_model_order = num_samples//2-2 # must be strictly less than num_samples/2
    
    magnitude_spectrum_fft = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(received_signal,axis=0, n=len(digital_freq_grid))/received_signal.shape[0],axes=0)))
    phase_spectrum_fft = np.unwrap(np.angle(np.fft.fftshift(np.fft.fft(received_signal,axis=0, n=len(digital_freq_grid))/received_signal.shape[0],axes=0)))
    
    spectrum = apes(received_signal, corr_mat_model_order, digital_freq_grid)
    magnitude_spectrum = np.abs(spectrum)
    phase_spectrum = np.unwrap(np.angle(spectrum))
    
    spectrum_iaa_nr = iaa_approx_nonrecursive(received_signal, digital_freq_grid) # non-recursive IAA
#    spectrum_iaa_nr = np.flipud(spectrum_iaa_nr)
    magnitude_spectrum_iaa_nr = np.abs(spectrum_iaa_nr)
    
    iterations = 10
    spectrum_iaa = iaa_recursive(received_signal, digital_freq_grid, iterations) # recursive IAA
#    spectrum_iaa = iaa_recursive_levinson_temp(received_signal, digital_freq_grid, iterations) # under debug
    
    magnitude_spectrum_iaa = np.abs(spectrum_iaa)    
    
    
    print('\nResolution improvement with respect to FFT = {}'.format(np.round(10*(fft_resol_fact/resol_fact))/10))
    
    apes_est_freq_ind = np.argsort(magnitude_spectrum,axis=0)[-num_sources::]
    estimated_complex_signal_amplitudes = spectrum[apes_est_freq_ind]
    
    iaa_nr_est_freq_ind = np.argsort(magnitude_spectrum_iaa_nr,axis=0)[-num_sources::]
    estimated_complex_signal_amplitudes_iaa_nr = spectrum_iaa_nr[iaa_nr_est_freq_ind]
    
    iaa_est_freq_ind = np.argsort(magnitude_spectrum_iaa,axis=0)[-num_sources::]
    estimated_complex_signal_amplitudes_iaa = spectrum_iaa[iaa_est_freq_ind]
    
    
    print('\nTrue Signal Amplitudes:{}, APES based Signal Amplitudes: {}'.format(np.abs(complex_signal_amplitudes),np.abs(estimated_complex_signal_amplitudes)))
    print('\nTrue Signal Phases(deg):{}, APES based Signal Phases(deg): {}'.format(np.angle(signal_phases)*180/np.pi,np.angle(estimated_complex_signal_amplitudes)*180/np.pi))
    
    print('\nTrue Signal Amplitudes:{}, Approx IAA non-recursive based Signal Amplitudes: {}'.format(np.abs(complex_signal_amplitudes),np.abs(estimated_complex_signal_amplitudes_iaa_nr)))
    print('\nTrue Signal Phases(deg):{}, Approx IAA non-recursive based Signal Phases(deg): {}'.format(np.angle(signal_phases)*180/np.pi,np.angle(estimated_complex_signal_amplitudes_iaa_nr)*180/np.pi))
    
    print('\nTrue Signal Amplitudes:{}, IAA recursive based Signal Amplitudes: {}'.format(np.abs(complex_signal_amplitudes),np.abs(estimated_complex_signal_amplitudes_iaa)))
    print('\nTrue Signal Phases(deg):{}, IAA recursive based Signal Phases(deg): {}'.format(np.angle(signal_phases)*180/np.pi,np.angle(estimated_complex_signal_amplitudes_iaa)*180/np.pi))
    
    
    plt.figure(3)
    plt.title('Magnitude Spectrum from APES')
    plt.plot(digital_freq_grid, 20*np.log10(magnitude_spectrum),alpha=0.7, label = 'APES')
    plt.plot(digital_freq_grid, 20*np.log10(magnitude_spectrum_iaa_nr),alpha=0.7, label = 'approx non-recursive IAA')
    plt.plot(digital_freq_grid, 20*np.log10(magnitude_spectrum_iaa),alpha=0.7, label = 'Recursive IAA')
    plt.plot(digital_freq_grid, magnitude_spectrum_fft, label = 'FFT')
    plt.vlines(-source_freq,-150,25, label = 'Ground truth')
    plt.xlabel('Digital Frequencies')
    plt.legend()    
    plt.grid(True)
    
#    plt.figure(4)
#    plt.title('Phase Spectrum from APES')
#    plt.plot(digital_freq_grid, phase_spectrum,'o-',alpha=0.7, label = 'Est PSD from APES')
#    plt.plot(digital_freq_grid, phase_spectrum_fft,'o-', label = 'FFT Power Specrum')
#    plt.vlines(-source_freq,-150,25, label = 'Ground truth')
#    plt.xlabel('Digital Frequencies')
#    plt.legend()


## IAA approx_non-recursive
if 0:
    plt.close('all')
    num_samples = 1024
    num_sources = 3
    object_snr = np.array([20,20,20])
    noise_power_db = -40 # Noise Power in dB
    noise_variance = 10**(noise_power_db/10)
    noise_sigma = np.sqrt(noise_variance)
    weights = noise_variance*10**(object_snr/10)
    source_freq = np.random.uniform(low=-np.pi, high=np.pi, size = num_sources)
    source_signals = np.matmul(np.exp(-1j*np.outer(np.arange(num_samples),source_freq)),weights[:,None])
    wgn_noise = np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape)
    received_signal = source_signals + wgn_noise
    corr_mat_model_order = 100 # must be strictly less than num_samples/2
    
    
    digital_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/(100*num_samples))    
    spectrum = iaa_approx_nonrecursive(received_signal, digital_freq_grid) # non-recursive IAA
    magnitude_spectrum = np.abs(spectrum)

    plt.figure(3)
    plt.title('Magnitude Spectrum')
    plt.plot(digital_freq_grid, 20*np.log10(magnitude_spectrum),alpha=1, label = 'Approx recursive IAA (compute heavy)')
    plt.vlines(-source_freq,-150,25, label = 'Ground truth')
    plt.xlabel('Digital Frequencies')
    plt.legend()    
    plt.grid(True)


### Approximate recursive IAA compute heavy
if 0:
    plt.close('all')
    num_samples = 512#1024
    num_sources = 3
    object_snr = np.array([20,20,20])
    noise_power_db = -40 # Noise Power in dB
    noise_variance = 10**(noise_power_db/10)
    noise_sigma = np.sqrt(noise_variance)
    weights = noise_variance*10**(object_snr/10)
    source_freq = np.random.uniform(low=-np.pi, high=np.pi, size = num_sources)
    source_signals = np.matmul(np.exp(-1j*np.outer(np.arange(num_samples),source_freq)),weights[:,None])
    wgn_noise = np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape)
    received_signal = source_signals + wgn_noise
    corr_mat_model_order = 100 # must be strictly less than num_samples/2
    
    
#    digital_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/(100*num_samples))    
#    spectrum = iaa_approx_nonrecursive(received_signal, digital_freq_grid) # non-recursive IAA
#    magnitude_spectrum = np.abs(spectrum)
    iterations = 10
    digital_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/(10*num_samples))    
    spectrum = iaa_approx_recursive_computeheavy(received_signal, digital_freq_grid,iterations) # non-recursive IAA
    magnitude_spectrum = np.abs(spectrum)
    plt.figure(3)
    plt.title('Magnitude Spectrum')
    plt.plot(digital_freq_grid, 20*np.log10(magnitude_spectrum),alpha=1, label = 'Approx non-recursive IAA')
    plt.vlines(-source_freq,-150,25, label = 'Ground truth')
    plt.xlabel('Digital Frequencies')
    plt.legend()    
    plt.grid(True)
    
    

### recursive IAA 
if 0:
    plt.close('all')
    num_samples = 1024
    num_sources = 3
    object_snr = np.array([20,20,20])
    noise_power_db = -40 # Noise Power in dB
    noise_variance = 10**(noise_power_db/10)
    noise_sigma = np.sqrt(noise_variance)
    weights = noise_variance*10**(object_snr/10)
    source_freq = np.random.uniform(low=-np.pi, high=np.pi, size = num_sources)
    source_signals = np.matmul(np.exp(-1j*np.outer(np.arange(num_samples),source_freq)),weights[:,None])
    wgn_noise = np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape)
    received_signal = source_signals + wgn_noise
    corr_mat_model_order = 100 # must be strictly less than num_samples/2
    
    
#    digital_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/(100*num_samples))    
#    spectrum = iaa_approx_nonrecursive(received_signal, digital_freq_grid) # non-recursive IAA
#    magnitude_spectrum = np.abs(spectrum)
    iterations = 10
    digital_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/(100*num_samples))    
    spectrum = iaa_recursive(received_signal, digital_freq_grid,iterations) # non-recursive IAA
    magnitude_spectrum = np.abs(spectrum)
    plt.figure(3)
    plt.title('Magnitude Spectrum')
    plt.plot(digital_freq_grid, 20*np.log10(magnitude_spectrum),alpha=1, label = 'recursive IAA')
    plt.vlines(-source_freq,-150,25, label = 'Ground truth')
    plt.xlabel('Digital Frequencies')
    plt.legend()    
    plt.grid(True)    




