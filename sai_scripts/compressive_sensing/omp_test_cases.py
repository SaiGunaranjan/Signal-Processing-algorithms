# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 00:31:07 2019

@author: Sai Gunaranjan Pelluri
"""


import numpy as np
import matplotlib.pyplot as plt
import compressive_sensing_lib as comp_sens





if 1:
    plt.close('all')
    num_rows = 208 # 16
    num_cols = 4241 # 356
    dictionary = np.random.randn(num_rows, num_cols)
    dictionary = dictionary/np.linalg.norm(dictionary,axis=0)
    sparsity = 5
    non_zero_ind = np.random.randint(num_cols, size = sparsity)
    x_vec = np.zeros((num_cols,1))
    x_vec[non_zero_ind,:] = 1
    y_vec = np.matmul(dictionary, x_vec)
    threshold = 1e-3
    x_vec_est, error_iter = comp_sens.OMP(dictionary, y_vec, threshold)
#    x_vec_est, error_iter = comp_sens.MP(dictionary, y_vec, threshold)
#    threshold = 1e1
#    y_vec = y_vec/np.linalg.norm(y_vec,axis=0)
#    x_vec_est, error_iter = comp_sens.MP_covariance(dictionary, y_vec, threshold)
    
    print('True Col Ind: ', non_zero_ind,  'Estimated Col Ind: ', np.nonzero(x_vec_est)[0])
    
    plt.figure(1)
    plt.plot(x_vec, 'o-')
    plt.plot(np.abs(x_vec_est), '*-')
    plt.grid(True)
    plt.xlabel('Column Index')
    plt.legend(['Ground Truth', 'Estimated from OMP'])
    #plt.figure(2)
    #plt.plot(np.array(error_iter),'o-')
    #plt.grid(True)
    #plt.xlabel('Iterations')
    #plt.ylabel('Error')