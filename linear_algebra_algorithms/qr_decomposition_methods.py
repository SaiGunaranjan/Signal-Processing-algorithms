# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:00:07 2024

@author: Sai Gunaranjan
"""

"""
In this script, I have implemented 2 flavours of QR decomposition namely the GramSchmidt orthogonalization
method and Householder tranformation method.

Any matrix (A) can be decomposed into an orthornormal matrix Q and an upper triangular matrix R. A = Q R
The QR decomposition finds its use in several algorithms like EVD, calculating inverse of a matrix, etc.
Hence it is a fundamental and important algorithm.

Gram Schmidt(GS) based QR decomposition:
The GS based QR decomposition constructs an orthonormal basis out of the columns of matrix A.
Now, to reconstruct the columns of A, all that is needed is the projections of columns of A
onto the orthonormal basis.
A = [e1 e2 e3..en] [<a1,e1>  <a2,e1>  <a3,e1> <a4,e1> ....
                    0        <a2,e2>  <a3,e2> <a4,e2> ....
                    0        0        <a3,e3> <a4,e3> ....
                    0        0        0       <a4,e4> ....
                    0        0        0       0
                    0        0        0       0
                    0        0        0       0
                                            ]

where e1, e2, e3 ..en are the orthonormal columns constructed out of the columns of A using GS factorization.
a1, a2, a3, ..an are the columns of A. <ak, ej> is the inner product between the kth column of A and
jth orthonormal column.
The matrix of orthonormal columns e1, e2, ..en is the Q matrix and the upper triangular matrix of
projections of columns of A onto columns of Q is the R matrix.
The GS factorization method to find columns of Q is straight forward.
GramSchmidt orthogonalization for QR decomposition is availabel at below link:
    https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/GramSchmidt.pdf



The Householder(HH) method of QR decomposition uses the Householder transformation.
Householder transformation:
The Householder transform of a point/vector x is defined wrt to a vector v.
The householder transformation gives the reflection of point x about the plane orthogonal to position vector v.
In a way, the HH transform is just rotating a vector x (without introducing any scaling whatsoever).
Hence, HH transform is a rotation matrix/orthonormal matrix.

Assume v is a unit vector.
<x,v> is the projection/inner product of x onto vector v.
<x,v>v is the projection vector of x in the direction of v.
x - <x,v>v is the projection vector of x onto the orthogonal complement of v.
x - <x,v>v - <x,v>v gives the reflection of point x about the plane orthogonal to v
So, the householder tranformation of a point x about a unit vector  is given by
Hv(x) = x - 2<x,v>v = x - 2(xTv)v = x - 2(vTx)v = x - 2 vvT x (since vTx is a scaler, it can be moved post v)
                                                = (I - 2 vvT)x
If v is not a unit vector, then we make it a unit vector first by normalizing with its norm.
So, the HH tranformation of point x wrt a vector v is given by:
    Hv(x) = (I - 2vvT/vTv)x

Theorem:
    Let x be a vector whose HH tranform we need to compute about a vector v. If v is chosen as a function
    of x as v = x + sign(x1)||x||e1, then, Hv(x) = -sign(x1)||x||e1,
    where, x is a non zero vector,
    x1 is the first element of vector x i.e x1 = x[0]
    ||x|| is the L2 norm of x
    e1 is the 1st vector of the standard bais i.e e1 = [1,0,0,0,0,0]
The proof is straightforward and also derived in the below link:
    https://math.byu.edu/~bakker/Math344/Lectures/M344Lec18.pdf

QR decomposition using Householder transformation essentially uses this theorem.
It involves transforming A matrix to a upper triangular matrix through successive multiplications with
different HH matrices. This means carefully applying the HH tranformation to each column one by one.
Apply HH tranform not to the entire column but a subset of the columns one after the other in an iterative manner.

Below is the algorithm to find the QR decomposition of a square matrix A using HH transformation:
    1. Take 1st column of matrix A and treat is as the x vector(of length n) in the above theorem. So, x = A[0,0::]
    2. Construct v using this x vector(1st column of A) as shown in above theorem.
    3. Construct the HH matrix Hv = (I - 2vvT/vTv).
    4. Apply the HH transformation on the matrix A i.e Hv A. This is same as subjecting each column
    of the A matrix to HH transform Hv. Since the 1st column of matrix A is used to construct the vector v,
    the HH tranform will convert the first column to the form -sign(A[0,0])||A[:,0]|| e1.
    Hence this vector will have only the 1st element as non zero and all other elements as 0.
    The resultant matrix multiplication will result in a matrix whose 1st column scaled version of [1,0,0,0,0].
    To eventually make the matrix upper traingular, we need to make the lower triangular part of the matrix to 0.
    5. For the second column, we need to make the entries from the 3rd row onwards to 0. So we construct HH matrix
    using A[1,1::]. So, the new x = A[1,1::] (of length n-1). Follow steps 2 and 3.
    Obtain the new HH matrix of shape n-1 x n-1
    6. Construct a new matrix as [I 0
                                  0 Hv]
    For the top row, I is the identity matrix and will be of shape 1 x 1 (so essentially 1 in this case). 0 matrix
    of shape 1 x n-1.
    Bottom row 0 matrix of shape n-1 x 1 and Hv of shape n-1 x n-1
    7. Multiply this with the matrix obtained in step 4.
    5. For the kth column, we need to make the entries from the k+1 row onwards to 0. So we construct HH matrix
    using A[k-1,k::]. So, the new x = A[k-1,k::] (of length n-k). Follow steps 2 and 3.
    Obtain the new HH matrix of shape n-k x n-k
    6. Construct a new matrix as B = [I 0
                                  0 Hv]
    For the top row, I is the identity matrix and will be of shape k x k. 0 matrix of shape k x n-k.
    Bottom row 0 matrix of shape n-k x k and Hv of shape n-k x n-k
    7. Multiply this with the matrix obtained in previous iteration.
    8. Keep following steps 5 to 7 till we finish all columns. The resultant matrix obtained is
    the required upper triangular matrix R.
    9. The Q matrix is the product of all the B matrices(conjugate transpose) obtained at each iteration.
    So Q = B1* B2* ...Bn*, where * indicates hermitiaon (conjugate transpose)


Householder transformation for QR decomposition:
    https://math.byu.edu/~bakker/Math344/Lectures/M344Lec18.pdf
Worked out example to show the Householder algorithm for QR decomposition is available in the below link:
    https://rpubs.com/aaronsc32/qr-decomposition-householder


I have extended the QR decomposition to both square and rectangular real matrices.
The GS based QR caters to both real and complex matrices while the HH based QR caters to only real matrices.
In the subsequent commits, I will extend the HH based QR to complex matrices as well.
"""

import numpy as np


def qr_gramschmidt(square_matrix):

    nrows, ncols = square_matrix.shape
    R = np.zeros((nrows,ncols),dtype=square_matrix.dtype)
    Q = np.zeros((nrows,nrows),dtype=square_matrix.dtype)
    numIter = min(nrows,ncols)
    for ele in range(numIter):
        """ Project an(nth column of A) onto e1, e2, ..en-1. (<an,e1>, <an,e2> ,.. <an,en-1>)"""
        R[:,ele] = Q.T.conj() @ square_matrix[:,ele]

        """ bn = an - (<an,e1>e1 + <an,e2>e2 +.. <an,en-1>en-1)"""
        temp = square_matrix[:,ele] - (Q @ R[:,ele])

        """ en = bn/||bn||"""
        Q[:,ele] = temp/np.linalg.norm(temp)

        """ <an,en> en is obtained in the previous step. Project an onto en"""
        R[ele,ele] = Q[:,ele].conj() @ square_matrix[:,ele]

    return Q, R

def qr_householder(square_matrix):

    nrows, ncols = square_matrix.shape
    R = square_matrix.copy()
    Q = np.eye(nrows)

    numIter = min(nrows,ncols)
    for ele in range(numIter):

        """ Step 5"""
        x = R[ele::,ele]
        e = np.zeros((nrows-ele),); e[0] = 1;

        """ Construct the vector"""
        v = (x + np.sign(x[0])*np.linalg.norm(x)*e)[:,None]

        """ Construct the Householder matrix"""
        houseHolderMatrix = np.eye(nrows-ele) - 2*((v @ v.T)/(v.T @ v))

        """ Step 6"""
        a1 = np.hstack((np.eye(ele),np.zeros((ele,nrows-ele))))
        a2 = np.hstack((np.zeros((nrows-ele,ele)), houseHolderMatrix))
        rotatMatrix = np.vstack((a1,a2))

        """ Step 7"""
        R = rotatMatrix @ R
        Q = Q @ rotatMatrix.T

    return Q, R


