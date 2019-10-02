# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:27:05 2019

@author: lenovo
"""

import numpy as np
#import sai_spectral_estimation as spec_est
import matplotlib.pyplot as plt
import time
import scipy.linalg
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
        

order = 1000
#x_vec = np.random.randint(1,100,order)
x_vec = np.random.randn(order) + 1j*np.random.randn(order) 
signal = np.random.randn(order) + 1j*1j*np.random.randn(order)
corr_vec = sts_correlate(signal[None,:])
corr_mat = vtoeplitz(corr_vec)[0,:,:]
y_vec = np.matmul(corr_mat,x_vec)
t1 = time.time()
x_vec_est = solve_levinson_durbin(corr_mat,y_vec)
t2 = time.time()
print('Time taken by my implementation: {} secs'.format(t2-t1))
t3 = time.time()
x_vec_est_scipy = scipy.linalg.solve_toeplitz(corr_vec.squeeze(),y_vec)
t4 = time.time()
print('Time taken by scipy package implementation: {} secs'.format(t4-t3))
plt.figure(1)
plt.subplot(121)
plt.title('Real Part')
plt.plot(np.real(x_vec),label='True x_vec')
plt.plot(np.real(x_vec_est_scipy),label='x_vec from scipy package')
plt.plot(np.real(x_vec_est),label='x_vec from my implementation')
plt.subplot(122)
plt.title('Imag Part')
plt.plot(np.imag(x_vec),label='True x_vec')
plt.plot(np.imag(x_vec_est_scipy),label='x_vec from scipy package')
plt.plot(np.imag(x_vec_est),label='x_vec from my implementation')
plt.legend()