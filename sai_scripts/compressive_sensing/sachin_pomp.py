# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 22:53:45 2019

@author: lenovo
"""

import numpy as np


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

def POMP_sai(A, bvec, num_iters, phi_max, tol, residual_threshold):
  n, m = A.shape
  x = np.zeros(m).astype(np.cfloat)
  xall = np.zeros((m, num_iters)).astype(np.cfloat)
  resall = np.zeros((n,num_iters+1)).astype(np.cfloat)
  res = bvec - np.dot(A,x[:,None])      # residual
  resn = np.linalg.norm(res)
  support = [];     # empty set to begin with
  Acopy = A.copy()
  for k in range(num_iters):
      if resn > tol:
#          print("------------------------")
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
#          print("Norm of residual after OMP step {}, :{}".format(k,np.linalg.norm(res)))
            
          # POMP steps
      
          res_n = res/np.linalg.norm(res)
          phi_star = np.arctan((np.linalg.norm(res) - tol)/np.linalg.norm(x,ord=1))
          phi_k = np.minimum(phi_max, phi_star)
          phi_k = np.squeeze(phi_k)
          ##{{ debug
#          zz = np.linalg.norm(res)-np.tan(phi_k[0])*np.linalg.norm(x,ord=1)
#          print("bperp upperbound {}".format(zz))
          # }}
#          print('Purturbing atoms by: {}'.format(phi_k[0]*180/np.pi))
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
          #bperp = res- res_n*np.tan(phi_k[0])*np.linalg.norm(x,ord=1)
#          res_pspace = np.dot(Orthspace_ProjectMat_perturbspace, bperp)
          #res_pspace = res - np.dot(A_sp*1/(np.cos(phi_k[0])),x[support])
#          print("Norm of residual in perturbed space: {}".format(np.linalg.norm(res_pspace)))
          ## }}
          #compute the new residual
          res = np.dot(np.eye(res.shape[0]) - np.dot(A_sp,A_sp.conj().T), bvec)
          resn = np.linalg.norm(res)
          
          #### for record keeping
#          print("Norm of residual after POMP step {}, :{}".format(k,resn))
          #print(support)
          #x[support] = np.real(np.dot(A_sp.conj().T,bvec))[:,0]
          
          x[support] = np.dot(A_sp.conj().T,bvec)[:,0]
          
          #print(x[support])
          xall[:,k] = x
          resall[:,k+1] = res.squeeze()          
          Acopy[:,support] = 0 #??
#          print('Iteration count:', k)
          
      else:
#          print(np.linalg.norm(res))
#          print(support)
#          print('exiting POMP, tolerance satisfied')
          break              
              
  residue = np.linalg.norm(resall[:,0:-1] - resall[:,1::],axis=0)
  x = np.abs(x)
  xall = np.abs(xall)
      #check for change in residual
  ind = np.where(residue < residual_threshold)[0]
  if ind.shape[0]==0: # if it is empty
      x = xall[:,0]
  else:
     x = xall[:,ind[0]]
  x_nonzInd = np.where(x!=0)[0]
          

         
  return x, xall, x_nonzInd
