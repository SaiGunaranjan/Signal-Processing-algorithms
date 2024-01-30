# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:40:58 2024

@author: Sai Gunaranjan
"""

"""
In this script, I have implemented the Eigen Value decomposition (EVD) using the iterative QR decomposition method.
    This method works if the matrix is a square matrix and in particular if it is a symmetric matrix
    Any matrix can be decomposed as a product of an orthonormal matrix Q and an upper traingular matrix R.
    There are several methods of computing this QR decomposition like Gram-Schmidt method, Householder method, etc.
    I will implement these methods in the subsequent commits. For now, lets assume we have the QR decompostion of a matrix A.
    The algorithm is as follows:
    Step 1: A = Q_0 R_0
    Step 2: Define a new matrix A1 = R_0 Q_0 (multiply Q and R in reverse order)
    Step 3: Now perform QR decompostion of A1. So we get A1 = Q_1 R_1
    Step 4: Again define a new matrix A2 = R_1 Q_1
    Step 5: Perform QR decomposition of A2 to get A2 = Q_2 R_2.
    .
    .
    .

    Keep repeating steps 3 through 5 for a few iterations.
    An = R_n-1 Q_n-1
    After a few iterations, An converges to an upper triangular matrix with the diagnal entries as the eigen values of A.
    If we keep on multiplying the Q matrices obtained at each iteration, then Q = Q_0 Q_1 Q_2 ....Q_n. The columns of Q matrix
    are the Eigen vectors of A.

    Now lets see the proof for this method.
    A = Q_0 R_0       --> QR decomposition of A
    R_0 = inv(Q_0) A  --> Since Q_0 is orthonormal, it is invertible and inv(Q_0) = hermitian(Q_0)
    Define A1 = R_0 Q_0
           = inv(Q_0) A Q_0  --> Since R_0 = inv(Q_0) A

    A1 = Q_1 R_1     --> QR decomposition of A1
    R_1 = inv(Q_1) A1
    Define A2 = R_1 Q_1
              = inv(Q_1) A1 Q_1
              = inv(Q_1) inv(Q_0) A Q_0 Q_1

    On repeating the above steps for n steps we get
    An = inv(Q_n-1) inv(Q_n-2) inv(Q_n-3).... inv(Q_1) inv(Q_0) A Q_0 Q_1 ... Q_n-3 Q_n-2 Q_n-1
       = herm(Q_n-1) herm(Q_n-2) herm(Q_n-3) ... herm(Q_1) herm(Q_0) A Q_0 Q_1 ... Q_n-3 Q_n-2 Q_n-1

    An converges to an upper traingular matrix
    Lets call herm(Q_n-1) herm(Q_n-2) herm(Q_n-3) ... herm(Q_1) = Qh
    and  Q_0 Q_1 ... Q_n-3 Q_n-2 Q_n-1 = Q

    So,
    An = Qh A Q
    Q is an orthonormal/unitary matrix since it is the product of several orthonormal matrices.

    Now, there's theorem called Schur's theorem which states that any
    square matrix is unitarily similar(similar matrices have same eigen values) to an upper traingular matrix i.e every square matrix can be written as
    A = UTU*, where T is an upper traingular matrix and U is an orthonormal matrix. So T = U* A U
    In our case, we have An = Qh A Q, where An is an upper traingular matrix.
    So the Eigen values of A are the same as the eigen values of An which are essentially the diagonal elements.
    Now, if A is also a symmetric matrix, then Q is the eigen vector matrix of A as well.


"""

import numpy as np

B = np.random.randn(9).reshape(3,3) # np.array([[1,2,3],[4,5,6], [7,8,9]])
A = B @ B.T
eigVal, eigVec = np.linalg.eig(A)

estEigVec = np.eye(3)
NUM_ITER = 6
for ele in range(NUM_ITER):
    Q, R = np.linalg.qr(A)
    A = R @ Q
    estEigVec = estEigVec @ Q
estEigVal = np.diag(A)

sorteigVals = np.sort(eigVal)
sortestEigVal = np.sort(estEigVal)

argeigVal = np.argsort(eigVal)
argestEigVal = np.argsort(estEigVal)

eigVecSort = eigVec[:,argeigVal]
estEigVecSort = estEigVec[:,argestEigVal]

print('Eigen Values using Lib = ', sorteigVals)
print('Eigen Values estimated using QR = ', sortestEigVal)

print('True Eigen Vectors = \n', eigVecSort)
print('Estimated Eigen Vectors = \n', estEigVecSort)

# print('\n Check for Orthonormality')
# print('U U* = \n', eigVec @ eigVec.T)
# print('Estimated U U* = \n', estEigVec @ estEigVec.T)