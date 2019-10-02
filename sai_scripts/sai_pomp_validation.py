# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 12:15:52 2019

@author: Sai Gunaranjan Pelluri
"""

import numpy as np
import scipy

def mutual_coherence(A):
    '''
        A: matrix of nxm
    '''
    # normalize the columns of A
    A_norm = np.linalg.norm(A, axis=0)
    An = A/A_norm[None,:]
    # construct Gram Matrix
    G = np.abs(np.dot(An.conj().T,An))
    np.fill_diagonal(G,0)
    mu = np.amax(G, axis=(0,1))
    return mu, G


def POMP_old(A, bvec, num_iters, phi_max, tol):
  n, m = A.shape
  #x = scipy.sparse.csc_matrix((m, 1), dtype=np.float) # all zeros sparse vector
  x = np.zeros(m).astype(np.cfloat)
  xall = np.zeros((m, num_iters)).astype(np.cfloat)
  resall = np.zeros(num_iters)
  res = bvec - np.dot(A,x[:,None])      # residual
  support = [];     # empty set to begin with
  
  for k in range(num_iters):
      if np.linalg.norm(res) > tol:
          print("------------------------")
          # choose the column of A that best describe the residual in terms of inner product
          next_atom = np.argmax(np.abs(np.dot(A.conj().T,res)))
          # add column to the set
          if next_atom not in support:
              support.append(next_atom) 
          
          # obtain LS soln of x based on updated S
          A_s = A[:, support]
          x_coeff = np.dot(scipy.linalg.pinv(A_s),bvec)
          x[support] = x_coeff[:,0]
          #x[x<0] = 0 # constrain te solution to have positive coefficient
          # update residual
          res = bvec - np.dot(A,x[:,None])
          print("Norm of residual after OMP step {}, :{}".format(k,np.linalg.norm(res)))
            
          # POMP steps
      
          res_n = res/np.linalg.norm(res)
          phi_star = np.arctan((np.linalg.norm(res) - tol)/np.linalg.norm(x,ord=1))
          phi_k = np.minimum(phi_max, phi_star)
          phi_k = np.squeeze(phi_k)
          ##{{ debug
          zz = np.linalg.norm(res)-np.tan(phi_k[0])*np.linalg.norm(x,ord=1)
          print("bperp upperbound {}".format(zz))
          # }}
          print('Purturbing atoms by: {}'.format(phi_k[0]*180/np.pi))
          A_sp = np.zeros_like(A_s)
          for i in range(A_s.shape[1]):
              # perturb the support vector
              #A_sp[:,i] = A_s[:,i]*np.cos(phi_k) + np.squeeze(res_n)*np.sign(x[support[i]])*np.sin(phi_k)
              A_sp[:,i] = A_s[:,i]*np.cos(phi_k[i]) + np.squeeze(res_n)*np.sign(x[support[i]])*np.sin(phi_k[i])
          
          ##{{ debug, compute residual bperp in perturbed space
          res_pspace = np.zeros_like(res)      
          ProjectMat_perturbspace = np.dot(A_sp,np.linalg.pinv(A_sp))
          Orthspace_ProjectMat_perturbspace = np.eye(ProjectMat_perturbspace.shape[0]) - ProjectMat_perturbspace
          bperp = res*(1 - 1/np.linalg.norm(res)*np.sum(np.abs(x[support])*np.tan(phi_k[support])))
          #bperp = res- res_n*np.tan(phi_k[0])*np.linalg.norm(x,ord=1)
          res_pspace = np.dot(Orthspace_ProjectMat_perturbspace, bperp)
          #res_pspace = res - np.dot(A_sp*1/(np.cos(phi_k[0])),x[support])
          print("Norm of residual in perturbed space: {}".format(np.linalg.norm(res_pspace)))
          ## }}
          #compute the new residual
          res = np.dot(np.eye(res.shape[0]) - np.dot(A_sp,A_sp.conj().T), bvec)
          resn = np.linalg.norm(res)
          
          #### for record keeping
          print("Norm of residual after POMP step {}, :{}".format(k,resn))
          #print(support)
          #x[support] = np.real(np.dot(A_sp.conj().T,bvec))[:,0]
          
          x[support] = np.dot(A_sp.conj().T,bvec)[:,0]
          
          #print(x[support])
          xall[:,k] = x
          resall[k] = resn
          
          
      else:
          print(np.linalg.norm(res))
          print(support)
          print('exiting POMP, tolerance satisfied')
          break;
                    
              
      
          
      #x[support] = np.real(np.dot(A_sp.conj().T,bvec))[:,0]   
          
  return np.abs(x), np.real(xall), resall



def POMP(A, bvec, num_iters, phi_max, tol):
  n, m = A.shape
  #x = scipy.sparse.csc_matrix((m, 1), dtype=np.float) # all zeros sparse vector
  x = np.zeros(m).astype(np.cfloat)
  xall = np.zeros((m, num_iters)).astype(np.cfloat)
  resall = np.zeros(num_iters)
  res = bvec - np.dot(A,x[:,None])      # residual
  support = [];     # empty set to begin with
  Acopy = A.copy()
  #l = list([211, 1496, 471, 707, 1460, 1657, 301])
  #l = list([211])
  #Acopy[:,l] = 0
  for k in range(num_iters):
      if np.linalg.norm(res) > tol:
          print("------------------------")
          # choose the column of A that best describe the residual in terms of inner product
          next_atom = np.argmax(np.abs(np.dot(Acopy.conj().T,res)))#??
          # add column to the set
          #if next_atom not in support and next_atom not in l:
          if next_atom not in support:    
              support.append(next_atom) 
          
          # obtain LS soln of x based on updated S
          A_s = A[:, support]
          x_coeff = np.dot(A_s.conj().T,bvec)#np.dot(scipy.linalg.pinv(A_s),bvec) #??
          x[support] = x_coeff[:,0]
          #x[x<0] = 0 # constrain te solution to have positive coefficient
          # update residual
          res = bvec - np.dot(A,x[:,None])
          print("Norm of residual after OMP step {}, :{}".format(k,np.linalg.norm(res)))
            
          # POMP steps
      
          res_n = res/np.linalg.norm(res)
          phi_star = np.arctan((np.linalg.norm(res) - tol)/np.linalg.norm(x,ord=1))
          phi_k = np.minimum(phi_max, phi_star)
          phi_k = np.squeeze(phi_k)
          ##{{ debug
          zz = np.linalg.norm(res)-np.tan(phi_k[0])*np.linalg.norm(x,ord=1)
          print("bperp upperbound {}".format(zz))
          # }}
          print('Purturbing atoms by: {}'.format(phi_k[0]*180/np.pi))
          A_sp = np.zeros_like(A_s)
          for i in range(A_s.shape[1]):
              # perturb the support vector
              #A_sp[:,i] = A_s[:,i]*np.cos(phi_k) + np.squeeze(res_n)*np.sign(x[support[i]])*np.sin(phi_k)
              A_sp[:,i] = A_s[:,i]*np.cos(phi_k[i]) + np.squeeze(res_n)*np.sign(x[support[i]])*np.sin(phi_k[i])
          
          ##{{ debug, compute residual bperp in perturbed space
#          res_pspace = np.zeros_like(res)      
#          ProjectMat_perturbspace = np.dot(A_sp,np.linalg.pinv(A_sp))
#          Orthspace_ProjectMat_perturbspace = np.eye(ProjectMat_perturbspace.shape[0]) - ProjectMat_perturbspace
#          bperp = res*(1 - 1/np.linalg.norm(res)*np.sum(np.abs(x[support])*np.tan(phi_k[support])))
#          #bperp = res- res_n*np.tan(phi_k[0])*np.linalg.norm(x,ord=1)
#          res_pspace = np.dot(Orthspace_ProjectMat_perturbspace, bperp)
#          #res_pspace = res - np.dot(A_sp*1/(np.cos(phi_k[0])),x[support])
#          print("Norm of residual in perturbed space: {}".format(np.linalg.norm(res_pspace)))
#          ## }}
          #compute the new residual
          res = np.dot(np.eye(res.shape[0]) - np.dot(A_sp,A_sp.conj().T), bvec)
          resn = np.linalg.norm(res)
          
          #### for record keeping
          print("Norm of residual after POMP step {}, :{}".format(k,resn))
          #print(support)
          #x[support] = np.real(np.dot(A_sp.conj().T,bvec))[:,0]
          
          x[support] = np.dot(A_sp.conj().T,bvec)[:,0]
          
          #print(x[support])
          xall[:,k] = x
          resall[k] = resn          
          
          Acopy[:,support] = 0 #??
          
      else:
          print(np.linalg.norm(res))
          print(support)
          print('exiting POMP, tolerance satisfied')
          break              
              
      
          
      #x[support] = np.real(np.dot(A_sp.conj().T,bvec))[:,0]   #??
          
  return np.abs(x), np.abs(xall), resall



def OMP(dictionary, y_vec, threshold):
    col_index = []
    basis = np.zeros((dictionary.shape[0],0)).astype('complex64')
    residue_mat = np.zeros((dictionary.shape[0],1)).astype('complex64')
    residue = y_vec
    x_vec_est = np.zeros(dictionary.shape[1]).astype('complex64')[:,None]
    error_iter = []
    res_err_cond = True
    while res_err_cond:
        ind = np.argmax(np.abs(np.matmul(dictionary.T, residue))) # Look for the column with maximum projection on the y/residue vector
        col_index.append(ind) # Store the column index
        basis = np.hstack((basis, dictionary[:,ind][:,None])) # Select that column which has the maximum correlation with the y/residue vector and append it to the columns obtained from previous iterations
        z_est = np.matmul(np.linalg.pinv(basis), y_vec) # compute the z_est such that z_est = pinv(basis)*y
        residue = y_vec - np.matmul(basis, z_est) # compute the residue/error as y-y^ where y is our measurement vector and y^ = basis*z_est(from previous step)
        residue_mat = np.hstack((residue_mat, residue)) # store the residue vector for each iteration(just to check how the error/residue is changing across ietrations)
        err = np.linalg.norm(residue_mat[:,-1] - residue_mat[:,-2]) # check the error in the residue across iterations to check if the residue is changing 
        res_err_cond =  err > threshold # check if the change in residue/error is below a particular threshold. Then stop
        error_iter.append(err)
#    valid_col_ind = np.sort(np.array(col_index))
#    z_est_sorted = z_est[np.argsort(np.array(col_index))]
#    x_vec_est[valid_col_ind] = z_est_sorted
    x_vec_est[col_index] = z_est
    return x_vec_est, error_iter



def MP(dictionary, y_vec, threshold):
    col_index = []
    residue_mat = np.zeros((dictionary.shape[0],1))
    residue = y_vec
    x_vec_est = np.zeros(dictionary.shape[1])[:,None]
    error_iter = []
    res_err_cond = True
    while res_err_cond:
        inner_prod = np.matmul(dictionary.T, residue)
        ind = np.argmax(np.abs(inner_prod)) # Look for the column with maximum projection on the y/residue vector
        col_index.append(ind) # Store the column index
        x_vec_est[ind] += inner_prod[ind] # Update the x vec index at each iteration . if same index repats then each iteration it gets added to the previous value
        chosen_atom =  dictionary[:,ind][:,None]
        residue -=  inner_prod[ind]*chosen_atom # compute the residue/error as y-y^ where y is our measurement vector and y^ = basis*z_est(from previous step)
        residue_mat = np.hstack((residue_mat, residue)) # store the residue vector for each iteration(just to check how the error/residue is changing across ietrations)
        err = np.linalg.norm(residue_mat[:,-1] - residue_mat[:,-2]) # check the error in the residue across iterations to check if the residue is changing 
        print('\n')
        print(err)
        res_err_cond =  err > threshold # check if the change in residue/error is below a particular threshold. Then stop
        error_iter.append(err)
    return x_vec_est, error_iter


















Naz = 81
Nel = 21
file_name_npy = 'b00059_calA_tdm_A.npy'
A_mat_3d = np.load(file_name_npy)
A_mat_3d = A_mat_3d[0::4,0::4,:]
A = A_mat_3d.reshape(-1,16).T
An=A/np.linalg.norm(A,axis=0)[None,:]

#num_rows = 16 # 16
#num_cols = 1701 # 356
#A = np.random.randn(num_rows, num_cols)

num_cols = A.shape[1]
sparsity = 3
non_zero_ind = np.random.randint(num_cols, size = sparsity)
x_vec = np.zeros((num_cols,1))
x_vec[non_zero_ind,:] = 1#non_zero_ind[:,None]
bvec = np.matmul(An, x_vec)
sigma = 0
bvec = bvec + sigma*(np.random.randn(bvec.shape[0],bvec.shape[1]) + 1j*np.random.randn(bvec.shape[0],bvec.shape[1]))
mu, G = mutual_coherence(A)
phi_max = 1/2*np.arccos(mu)*np.ones(A.shape[1])
x, temp1, temp2 = POMP(An, bvec, num_iters=sparsity, phi_max=phi_max, tol=1e-3)
#x, temp1, temp2 = POMP_old(An, bvec, num_iters=10, phi_max=phi_max, tol=1e-3)
inx1d= np.where(x !=0)[0]#[0:2]
sortinx= np.argsort(x[inx1d])[::-1]

print('\n\n\nTrue col ind:', non_zero_ind, 'Est col ind from POMP:', inx1d[sortinx])

threshold = 1e-3
x_vec_est, error_iter = OMP(An, bvec, threshold)
#x_vec_est, error_iter = MP(dictionary, y_vec, threshold)

print('\nTrue Col Ind: ', non_zero_ind,  'Estimated Col Ind OMP: ', np.nonzero(x_vec_est)[0])

