# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 21:06:12 2019

@author: Sai Gunaranjan Pelluri
"""
import numpy as np



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
    for ele in np.arange(signal_length-corr_mat_model_order+1): # model order dictates the length of the auto-correlation matrix
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
    """ GhA can be computed in 2 ways:
        1. As a correlation with a vandermonde matrix
        2. Oversampled FFT of each of the noise subspace vectors
        Both 1 and 2 are essentially one and the same. But 1 is compute heavy in terms of MACS while 2 is more FFT friendly

    """
    if 0:
        GhA = np.matmul(noise_subspace.T.conj(),vandermonde_matrix) #G*A essentially projects the vandermond matrix (which spans the signal subspace) on the noise subspace
    if 1:
        GhA = np.fft.fftshift(np.fft.fft(noise_subspace.T.conj(),n=len(digital_freq_grid),axis=1),axes=(1,)) # Method 2

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
    '''corr_mat_model_order : must be strictly less than half the signal length'''
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
    '''corr_mat_model_order : must be strictly less than half the signal length'''
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
        Rinv_y = np.matmul(auto_corr_matrix_inv, received_signal)
        """ Ah_Rinv_y can be computed in 2 ways:
            1. Using matrix multiplication/point wise multiplication + sum
            2. Oversampled FFT of Rinv_y followed by fft shift
        Similarly, Rinv_A can be computed in 2 ways:
            1. matrix multiplication of R_inv and Vandermonde matrix
            2. Oversampled FFT of each row of R_inv followed by FFT shift along the column dimension and
            then a flip from left to right. This flip is required because in method 1, we are simply
            correlating each row of R_inv with a sinusoid vector whose frequency is going from -pi to +pi
            from left most column to right most column. This means we are actually obtaining the frequency
            strengths from +pi to -pi(Since FFT has an implicit conjugate sign in the kernel).
            Hence to match the output from method 1, we need to do a flip from left to right.
            3. Since the frequency grid points are from -pi to pi, we can use IFFT instead of FFT+FLIPLR.
            But the scaling with the IFFT needs to be handled.

        Ah_Rinv_A HAS to be done as a point wise multiplication and sum. It cannot be cast as FFTs.
        Method 2 in both the cases is compute efficient since we cast the beam former multiplications as FFTs.
        """
        if 0:
            Ah_Rinv_y = np.sum(vandermonde_matrix.conj()*Rinv_y,axis=0) # Method 1 for Ah_Rinv_y
            Rinv_A = np.matmul(auto_corr_matrix_inv,vandermonde_matrix) # Method 1 for Rinv_A
        if 1:
            Ah_Rinv_y = np.fft.fftshift((np.fft.fft(Rinv_y,axis=0,n=num_freq_grid_points)),axes=(0,)).squeeze() # Method 2 for Ah_Rinv_y
            Rinv_A = np.fliplr(np.fft.fftshift(np.fft.fft(auto_corr_matrix_inv,axis=1,n=num_freq_grid_points),axes=(1,))) # Method 2 for Rinv_A - Use FFT,FFTSHIFT,fliplr
            # Rinv_A = np.fft.fftshift(np.fft.ifft(auto_corr_matrix_inv,axis=1,n=num_freq_grid_points),axes=(1,)) # Method 3 for Rinv_A - Use IFFT, FFTSHIFT. Avoids fliplr

        Ah_Rinv_A = np.sum(vandermonde_matrix.conj()*Rinv_A,axis=0)
        spectrum = Ah_Rinv_y/Ah_Rinv_A
        # print(iter_num)
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


"""

Spatially Variant Apodization

In this script, I have implemented a technique called Spatially Variant Apodization (SVA). SVA is a like for like replacement
for a windowed FFT. A typical windowed FFT is characterized by main lobe width and side lobe level.
Rectangular window has very good main lobe width but poor SLLs (13 dBc). Hanning window has poor main lobe width but good side
SLLs (31 dBc). SVA is a technique which combines the best of both these windows. It gives the main lobe width of a rectangular window
while giving the SLL of a Hanning window. Thus it gives the best of both worlds. The working principle of SVA is as follows:

The SVA attemps to select a different window function for each frequency. Let us understand this more closely.
A raised cosine window is a generic window which is controlled by an alpha parameter. The functional form of the window is
1-2*alpha*cos(2*pi/N*n), where 'alpha' lies in [0,0.5], 'N' is the number of samples and 'n' varies from 0 to N-1.
When 'alpha' = 0, the raised cosine window becomes a rectangular window. When 'alpha' = 0.5, the window becomes a hanning window.
By varying alpha from 0 to 0.5 in fine steps, we can create multiple variants of the raised cosine window.
A brute force version of the SVA is as follows. Subject the signal to each of these windows and perform an FFT on each of
the windowed signals. This results in several FFTs of the signal each with a different window. Now, compute the psd/magnitude square
of each of these FFTs. To obtain the final psd/spectrum, take minimum along the window axis. This results in a spectrum which has
minimum energy at each frequency point across all window function. In essence, we are selecting a window function which generates
the minimum energy at each frequency point. Since, we are selecting munimum energy at each frequency point (across all windows),
this will result in minimum SLLs as well as best main lobe width. But this brute force method requires heavy compute since
we need to generate several windows and multiply the signal also with each of these windows and compute FFT on each of these signals.
Finally we also need to take a minimum for each of the frequency point. This is serioulsy heavy compute as well as memory.
This can be circumvented by implementing the optimized version of the SVA. For this, we need to take a closer look at the structure
of the raised cosine window. The functional form of the window is w[n] = 1-2*alpha*cos(2*pi/N*n).
If we assume the signal of interest is x[n], then the windowed signal is y[n] = x[n]w[n]. Now multiplication in time domain is
equivalent to convolution in the frequency domain. So if we were to analyse the windowed signal in the Fourier domain,
we get Y(w) = X(w) * W(w), where * denotes convolution operation. Now W(w) = 2pi * ( delta(w) - alpha*[delta(w-2*pi/N) + delta(w+2*pi/N)]).
So,  Y(w) = 2pi * (X(w) - alpha*[X(w-2pi/N) + X(w+2pi/N)]). Now, the optimum value of alpha(at each frequency) is chosen by
minimizing the magnitude squared of Y(w) over all alpha lying between 0 to 0.5. This is a constrained optimization problem.
Solving this, we obtain a closed form expression for alpha at each frequency point omega.
Hence alpha is a function of the omega chosen. The other details of the derivation are available in the paper link below:
https://saigunaranjan.atlassian.net/wiki/spaces/RM/pages/22249477/Spatially+Variant+Apodization

After obtaining a closed form expression for alpha and substituting in the expression for Y(w), we obtain the spectrum for SVA.
This method of computing the SVA spectrum is computationally light. As evident from the closed form expression for alpha,
the only major compute (apart from the FFT computation of x[n]) is an N point division for an N point oversampled FFT.
So there are N complex divisions corresponding to each of the N frequency points.
I have implemented both the brute force method as well as the optimized method of SVA and have generated the results.
Both thse methods seem to be closely matching. The results are on expected lines with the SVA offering the best SLLS as well as
main lobe width.
"""


def spatially_variant_apodization_bruteforce(received_signal,numFFTOSR):

    """
    received_signal should be a column vector e.g: 32 x 1

    """
    num_samples = received_signal.shape[0]
    alpha = np.linspace(0,0.5,100)
    svaWindow = 1 - 2*alpha[None,:]*np.cos(2*np.pi*np.arange(num_samples)[:,None]/num_samples)
    signalWindowed = received_signal * svaWindow
    svaSignalFFT = np.fft.fft(signalWindowed,n=numFFTOSR,axis=0)/num_samples
    svaSignalFFTShift = np.fft.fftshift(svaSignalFFT,axes=(0,))
    svaSignalPsd = np.abs(svaSignalFFTShift)**2
    svaSignalPsdNormalized = svaSignalPsd/np.amax(svaSignalPsd,axis=0)[None,:]
    svaSpectralEstimator = np.amin(svaSignalPsdNormalized,axis=1)
    svaSpectralEstimatordB_unoptimal = 10*np.log10(svaSpectralEstimator)

    return svaSpectralEstimatordB_unoptimal


def spatially_variant_apodization_optimized(received_signal, osrFact):


    num_samples = received_signal.shape[0]
    received_signal_sva = np.squeeze(received_signal)
    numFFTOSR = osrFact*num_samples
    signalFFT = np.fft.fft(received_signal_sva,n=numFFTOSR,axis=0)/num_samples # Is normalization required here for sva
    Xk = signalFFT
    kmKInd = np.arange(0,numFFTOSR) - osrFact
    kmKInd[kmKInd<0] += numFFTOSR
    XkmK = Xk[kmKInd]
    kpKInd = np.arange(0,numFFTOSR) + osrFact
    kpKInd[kpKInd>numFFTOSR-1] -= numFFTOSR
    XkpK = Xk[kpKInd]
    alphaK = np.real(Xk/(XkmK+XkpK))
    alphaK[alphaK<0] = 0
    alphaK[alphaK>0.5] = 0.5
    svaspectrum = Xk - alphaK*(XkmK+XkpK)
    svaOptimalComplexSpectrumfftshifted = np.fft.fftshift(svaspectrum)
    svaOptimalMagSpectrumdB = 20*np.log10(np.abs(svaOptimalComplexSpectrumfftshifted))
    svaOptimalMagSpectrumdB -= np.amax(svaOptimalMagSpectrumdB)

    return svaOptimalComplexSpectrumfftshifted, svaOptimalMagSpectrumdB






