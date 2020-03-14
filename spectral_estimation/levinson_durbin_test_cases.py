# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:27:05 2019

@author: lenovo
"""

import numpy as np
from spectral_estimation_lib import solve_levinson_durbin, sts_correlate, vtoeplitz
import matplotlib.pyplot as plt
from time import time
import scipy.linalg


        

order = 1000
#x_vec = np.random.randint(1,100,order)
x_vec = np.random.randn(order) + 1j*np.random.randn(order) 
signal = np.random.randn(order) + 1j*1j*np.random.randn(order)
corr_vec = sts_correlate(signal[None,:])
corr_mat = vtoeplitz(corr_vec)[0,:,:]
y_vec = np.matmul(corr_mat,x_vec)
t1 = time()
x_vec_est = solve_levinson_durbin(corr_mat,y_vec)
t2 = time()
print('Time taken by my implementation: {0:.1f} ms'.format((t2-t1)*1000))
t3 = time()
x_vec_est_scipy = scipy.linalg.solve_toeplitz(corr_vec.squeeze(),y_vec)
t4 = time()
print('Time taken by scipy package implementation: {0:.1f} ms'.format((t4-t3)*1000))
plt.figure(1)
plt.subplot(221)
plt.title('Real Part')
plt.plot(np.real(x_vec),'-o',label='True x_vec')
plt.plot(np.real(x_vec_est_scipy),'-o',label='x_vec from scipy package')
plt.plot(np.real(x_vec_est),'-o',label='x_vec from my implementation');
plt.grid(True)
plt.subplot(222)
plt.title('Imag Part')
plt.plot(np.imag(x_vec),'-o',label='True x_vec')
plt.plot(np.imag(x_vec_est_scipy),'-o',label='x_vec from scipy package')
plt.plot(np.imag(x_vec_est),'-o',label='x_vec from my implementation')
plt.legend();
plt.grid(True)
plt.subplot(223)
plt.title('Error in Real Part of x_vec')
plt.plot(np.real(x_vec)-np.real(x_vec_est_scipy),'-o',label='error from scipy package')
plt.plot(np.real(x_vec)-np.real(x_vec_est),'-o',label='error from my implementation');
plt.grid(True)
plt.subplot(224)
plt.title('Error in Imag Part of x_vec')
plt.plot(np.imag(x_vec)-np.imag(x_vec_est_scipy),'-o',label='error from scipy package')
plt.plot(np.imag(x_vec)-np.imag(x_vec_est),'-o',label='error from my implementation')
plt.legend();
plt.grid(True)