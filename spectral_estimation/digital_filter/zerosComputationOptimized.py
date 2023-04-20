# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 13:38:47 2023

@author: Sai Gunaranjan
"""


import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

def generateVandermondeSuperSet(numOrigSamp, numSamp):

    """
    inputs:
        1. numOrigSamp: original signal scale (say 168)
        2. numSamp: number of samples of the complex sinusoid (say 14)

    Output:
        vandermondeMatrixSuperSet: Vandermonde matrix superset of complex sinusoids with #rows = numSamp and #cols = numOrigSamp.

    The vandermonde matrix is also fft shifted so that the corresponding columns can be accessed using a unipolar linear indexing
    """
    vandermondeMatrixSuperSet = np.exp(1j*2*np.pi/numOrigSamp*(np.arange(numSamp)[:,None]*np.arange(-numOrigSamp//2, numOrigSamp//2)[None,:]))
    vandermondeMatrixSuperSet = np.fft.fftshift(vandermondeMatrixSuperSet, axes=1)

    return vandermondeMatrixSuperSet


def filterResponse2zeros(vandermondeMatrixSuperSet):

    """
    This function takes in a Vandermonde matrix and generates the zeros of each column of the matrix.
    This function can be used to generate the zeros of a vandermonde matrix upfront and then store them
    inputs:
        1. vandermondeMatrixSuperSet: Matrix of sinusoids stacked as columns

    outputs:
        1. zerosFilterResponseSuperSet: Zeros of each column of filterResponseSuperSet stacked again as columns.
        Note: # rows of zerosFilterResponseSuperSet = # of rows of filterResponseSuperSet -1 (Since the number of zeros is 1 less than the filter tap length)
        This function uses the inbuilt scipy.signal.tf2zpk
    """

    filterResponseSuperSet = np.conj(vandermondeMatrixSuperSet) ## This has to have a conjugation effect to enhance the true tone
    numResponses = filterResponseSuperSet.shape[1]
    numZeros = filterResponseSuperSet.shape[0] - 1
    zerosFilterResponseSuperSet = np.zeros((numZeros,numResponses),dtype=np.complex64)
    for ele in np.arange(numResponses):
        zerosFilterResponseSuperSet[:,ele], _, _ = sig.tf2zpk(filterResponseSuperSet[:,ele], 1)

    return zerosFilterResponseSuperSet

nDopp = 1024
Nramp_2 = 14
vandermondeMatrixSuperSet = (generateVandermondeSuperSet(nDopp, Nramp_2)).astype('complex64')
zerosFilterResponseSuperSet = filterResponse2zeros(vandermondeMatrixSuperSet)

zerosSimpleCompute = (2*np.pi/nDopp)*np.arange(nDopp)[None,:] + 2*np.pi*np.arange(1,Nramp_2)[:,None]/Nramp_2
# zerosSimpleComputePhasor = np.exp(-1j*zerosSimpleCompute)
zerosSimpleComputePhasor = np.cos(zerosSimpleCompute) - 1j*np.sin(zerosSimpleCompute)

ind = 897

plt.figure(1,figsize=(20,10),dpi=200)
plt.title('LUT zeros vs dynamically computed zeros using simple formula')
plt.plot(zerosFilterResponseSuperSet[:,ind].real, zerosFilterResponseSuperSet[:,ind].imag, 'o', ms=7, color = 'b', label='Solved zeros')
plt.plot(zerosSimpleComputePhasor[:,ind].real, zerosSimpleComputePhasor[:,ind].imag, 'o', ms=10, color='red', alpha=0.5, label='Computed Zeros')
plt.axis('equal')
plt.grid(True)
plt.xlabel('Real')
plt.ylabel('Imag')
plt.legend()

# plt.figure(1,figsize=(20,10),dpi=200)
# # plt.title('Zeros of 1st row of pseudo inverse matrix')
# plt.plot(zerosFilterResponseSuperSet[:,ind].real, zerosFilterResponseSuperSet[:,ind].imag, 'o', fillstyle='none', ms=7, color = 'b', lw=14, alpha=0.5, label='Solved zeros')
# plt.plot(zerosSimpleComputePhasor[:,ind].real, zerosSimpleComputePhasor[:,ind].imag, 'o', fillstyle='none', ms=7, color='red', label='Computed Zeros')
# plt.axis('equal')
# plt.grid(True)
# plt.xlabel('Real')
# plt.ylabel('Imag')
# plt.legend()