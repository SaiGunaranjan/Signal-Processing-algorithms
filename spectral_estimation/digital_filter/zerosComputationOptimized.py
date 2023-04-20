# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 13:38:47 2023

@author: Sai Gunaranjan
"""


"""
Compute Zeros of a sinusoid transfer function in an optimized manner

In this script, I have developed a simple method of computing the zeros of a sinusoid transfer function.
Ideally, in order to find the zeros of an FIR transfer function, we need to compute Z transform of the transfer function/impulse response
and then solve for the roots of the polynomial in Z inverse to obtain the roots/zeros of the transfer function.
However, if the signal/impulse response/transfer function is a sinusoid, we don't need to solve for the roots
of a polynomial(which is a compute heavy operation). Instead, we can find them in a simple manner.
We know that if we have a sinusoidal signal of digital frequency w and of N samples, if we perform an FFT,
the zeros are separated by 2*pi/N. These zeros/nulls will be visible with a simple N point FFT only for those sinusoidal signals
whose digital frequency is an integer multiple of 2*pi/N. For sinusoidal signals whose digital frequency is not an integer multiple
of 2*pi/N, a simple N point FFT will not reveal the nulls or zeros. To visualize the zeros/nulls, we need to perform an over sampled FFT.
This is because, if the digital frequency doesn't fall on a bin, then the zeros/nulls which occur at a separation of 2*pi/N from
the signal peak are also not evaluted when a simple N point FFT is performed. However, when an oversampled FFT is performed,
we are essentailly evaluating the spectrum on a much finer grid and hence we will inevitably end up evaluating the spectrum at
a frequency point which is 2*pi/N away from the signal peak as well. Then we will start observing the nulls/zeros.
Since this is establised, to compute the zeros for the sinusoidal transfer function of N samples, all we need to do is
to add integer multiples of 2*pi/N to the signal digital frequency in order to obtain the zeros.
If the signal has a length of N, then there will be N-1 zeros. So, if the sinusoidal signal has a digital frequency w, then,
 the zeros will be w + 2*pi/N * n, where n goes from (1,N-1). If n is 0 or N, then we land on the signal frequency and hence
 these points are avoided. Prevously, to compute the zeros of even a sinusoidal transfer function, I was calling the tf2zpk function
 from the scipy.signal library. This function essentially computes the Z transform of the signal and then solves for the roots of
 the polynomial. Now, I have understood the structure of a sinusoidal sigal and hence compute the zeros in a trivial manner.
 This reduces compute significant. This is a new trick I have realized !

Note: This trick is valid only for sinusoidal transfer functions (single sinusoidal signal of N samples)

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

ind = 897 # Showing the result with one index of the vandermonde matrix. But the result is valid for any sinusoidal signal

plt.figure(1,figsize=(20,10),dpi=200)
plt.title('LUT zeros vs dynamically computed zeros using simple formula')
plt.plot(zerosFilterResponseSuperSet[:,ind].real, zerosFilterResponseSuperSet[:,ind].imag, 'o', ms=7, color = 'b', label='Solved zeros')
plt.plot(zerosSimpleComputePhasor[:,ind].real, zerosSimpleComputePhasor[:,ind].imag, 'o', ms=10, color='red', alpha=0.5, label='Computed Zeros')
plt.axis('equal')
plt.grid(True)
plt.xlabel('Real')
plt.ylabel('Imag')
plt.legend()

""" I'm not computing the error between the zeros compued by library function and my method, since the order of the zeros might differ
in the 2 methods"""