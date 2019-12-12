# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 00:31:07 2019

@author: Sai Gunaranjan Pelluri
"""


import numpy as np
import matplotlib.pyplot as plt
#from scipy.linalg import hadamard

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


def MP(dictionary_matrix, y_vec, threshold):
    dictionary = dictionary_matrix.copy()
    dictionary = dictionary/np.linalg.norm(dictionary,axis=0)
    col_index = []
    residue_mat = np.zeros((dictionary.shape[0],1)).astype('complex64')
    residue = y_vec.copy()
    x_vec_est = np.zeros(dictionary.shape[1]).astype('complex64')[:,None]
    error_iter = []
    res_err_cond = True
    while res_err_cond:
        inner_prod = np.matmul(np.conj(dictionary.T), residue)
        ind = np.argmax(np.abs(inner_prod)) # Look for the column with maximum projection on the y/residue vector
        col_index.append(ind) # Store the column index
        x_vec_est[ind] += inner_prod[ind] # Update the x vec index at each iteration . if same index repats then each iteration it gets added to the previous value
        chosen_atom =  dictionary[:,ind][:,None]
        residue -=  inner_prod[ind]*chosen_atom # compute the residue/error as y-y^ where y is our measurement vector and y^ = basis*z_est(from previous step)
        residue_mat = np.hstack((residue_mat, residue)) # store the residue vector for each iteration(just to check how the error/residue is changing across ietrations)
        err = np.linalg.norm(residue_mat[:,-1] - residue_mat[:,-2]) # check the error in the residue across iterations to check if the residue is changing 
#        print('\n')
#        print(err)
        res_err_cond =  err > threshold # check if the change in residue/error is below a particular threshold. Then stop
        error_iter.append(err)
    return x_vec_est, error_iter

def OMP(dictionary_matrix, y_vec, threshold):
    dictionary = dictionary_matrix.copy()
    dictionary = dictionary/np.linalg.norm(dictionary,axis=0)
    col_index = []
    basis = np.zeros((dictionary.shape[0],0)).astype('complex64')
    residue_mat = np.zeros((dictionary.shape[0],1)).astype('complex64')
    residue = y_vec.copy()
    x_vec_est = np.zeros(dictionary.shape[1]).astype('complex64')[:,None]
    error_iter = []
    res_err_cond = True
    count = 1
    while res_err_cond:
        ind = np.argmax(np.abs(np.matmul(np.conj(dictionary.T), residue))) # Look for the column with maximum projection on the y/residue vector
        col_index.append(ind) # Store the column index
        basis = np.hstack((basis, dictionary[:,ind][:,None])) # Select that column which has the maximum correlation with the y/residue vector and append it to the columns obtained from previous iterations
        z_est = np.matmul(np.linalg.pinv(basis), y_vec) # compute the z_est such that z_est = pinv(basis)*y
        residue = y_vec - np.matmul(basis, z_est) # compute the residue/error as y-y^ where y is our measurement vector and y^ = basis*z_est(from previous step)
        residue_mat = np.hstack((residue_mat, residue)) # store the residue vector for each iteration(just to check how the error/residue is changing across ietrations)
        err = np.linalg.norm(residue_mat[:,-1] - residue_mat[:,-2]) # check the error in the residue across iterations to check if the residue is changing 
        res_err_cond =  err > threshold # check if the change in residue/error is below a particular threshold. Then stop
        error_iter.append(err)
        print(count)
        count+=1
#    valid_col_ind = np.sort(np.array(col_index))
#    z_est_sorted = z_est[np.argsort(np.array(col_index))]
#    x_vec_est[valid_col_ind] = z_est_sorted
    x_vec_est[col_index] = z_est
    return x_vec_est, error_iter



def MP_covariance(dictionary_matrix, y_vec, threshold):
    dictionary = dictionary_matrix.copy()
#    dictionary = dictionary/np.linalg.norm(dictionary,axis=0)
    num_cols = dictionary.shape[1]
    num_rows = dictionary.shape[0]
    dict_cov_num_rows = (num_rows*(num_rows+1))//2
    dict_cov= np.zeros((dict_cov_num_rows,num_cols)).astype('complex64')
    for ele in np.arange(num_cols):
        outer_prod = dictionary[:,ele][:,None]*np.conj(dictionary[:,ele][None,:])
        dict_cov[:,ele] = outer_prod[np.triu_indices(num_rows)]
    dict_cov = dict_cov/np.linalg.norm(dict_cov,axis=0)
    y_vec_copy = y_vec.copy()
    y_vec_corr = sts_correlate(y_vec_copy.T)
    y_vec_toeplitz = vtoeplitz(y_vec_corr) # Create a toeplitz matrix which is variant of the Auto-correlation matrix
    y_vec_toeplitz = y_vec_toeplitz[0,:,:]
    new_y_vec = y_vec_toeplitz[np.triu_indices(num_rows)][:,None]
    
    col_index = []
    residue_mat = np.zeros((dict_cov.shape[0],1)).astype('complex64')
    residue = new_y_vec.copy()
    x_vec_est = np.zeros(dict_cov.shape[1]).astype('complex64')[:,None]
    error_iter = []
    res_err_cond = True
    while res_err_cond:
        inner_prod = np.matmul(np.conj(dict_cov.T), residue)
        ind = np.argmax(np.abs(inner_prod)) # Look for the column with maximum projection on the y/residue vector
        col_index.append(ind) # Store the column index
        x_vec_est[ind] += inner_prod[ind] # Update the x vec index at each iteration . if same index repats then each iteration it gets added to the previous value
        chosen_atom =  dict_cov[:,ind][:,None]
        residue -=  inner_prod[ind]*chosen_atom # compute the residue/error as y-y^ where y is our measurement vector and y^ = basis*z_est(from previous step)
        residue_mat = np.hstack((residue_mat, residue)) # store the residue vector for each iteration(just to check how the error/residue is changing across ietrations)
        err = np.linalg.norm(residue_mat[:,-1] - residue_mat[:,-2]) # check the error in the residue across iterations to check if the residue is changing 
#        print('\n')
#        print(err)
        res_err_cond =  err > threshold # check if the change in residue/error is below a particular threshold. Then stop
        error_iter.append(err)
    return x_vec_est, error_iter


def mutual_coherence(dictionary):
    dictionary = dictionary/np.linalg.norm(dictionary,axis=0)
    cross_corr_mat = np.matmul(dictionary.T.conj(),dictionary)
    np.fill_diagonal(cross_corr_mat,0)
    mu = np.amax(cross_corr_mat,axis=(0,1))
    return mu
    
    



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
    x_vec_est, error_iter = OMP(dictionary, y_vec, threshold)
#    x_vec_est, error_iter = MP(dictionary, y_vec, threshold)
#    threshold = 1e1
#    y_vec = y_vec/np.linalg.norm(y_vec,axis=0)
#    x_vec_est, error_iter = MP_covariance(dictionary, y_vec, threshold)
    
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