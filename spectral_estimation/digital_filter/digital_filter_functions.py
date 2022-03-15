# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:06:36 2022

@author: saiguna
"""


import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt


def angular_distance(theta_1, theta_2, mod=2*np.pi):

    """
    This function is used to get the difference of two arrays on a modulo scale. ex: 355 is closer to 0 deg than 330 degress on a modulo scale of 360
    inputs:
        theta_1: Array 1
        theta_2: Array 2

    outputs:
        1. minVal: Minimum Difference value between every element of theta_1 and corresponding element of theta_2
    """
    difference = np.abs(theta_1 % mod - theta_2 % mod)
    minVal = np.minimum(difference, mod - difference)

    return minVal

def min_angular_distance_index(theta_1, theta_2, mod=2*np.pi):

    """
    This function is used to get the indices(on theta_2 array) of the closest values from
    the array theta_1 to array theta_2 on a modulo scale. ex: 355 is closer to 0 deg than 330 degress on a modulo scale of 360
    inputs:
        theta_1: Array 1
        theta_2: Array 2 whose indices for which the values in theta_2 are closest to theta_1 on a modulo scale

    outputs:
        1. closestInd: Indices of values in array theta_2 closest to values in array theta_1
        2. minVal: Minimum Difference value between every element of thets_1 and every element of theta_2
    """
    difference = np.abs(theta_1[:,None] % mod - theta_2[None,:] % mod)
    minVal = np.minimum(difference, mod - difference)
    closestInd = np.argmin(minVal,axis=1)
    return closestInd, minVal

def zeros2tf_conv(zeros_sig):

    """
    This function takes as input the signal zeros and outputs the FIR filter tap coefficients using the method of successive convolution
    inputs:
        1. zeros_sig: complex zeros of the signal

    outputs:
        1. ImpResponse: Impulse response obtained by successive convolution of (# zeros) of 2 tap FIR filters.
        The length of ImpResponse = # Zeros + 1

    Concept:
        An FIR system of N zeros can be modelled as N 2 tap cascaded FIR filters whose coefficients are h_i[n] = [1, -z_i]
        where h_i[n] is the impulse response of the ith filter and z_i is the ith zero of the system.
        Since the overall Z tranform is a product of the form (1-z_1*z^-1) * (1-z_2*z^-1) * ... ((1-z_N*z^-1)), the overall
        impulse response is given by h[n] = h1[n] conv h2[n] cov ....[hN[n]]
        Hence, here in this function, we find the final imuplse response of the system by successive convolution
    """

    numZeros = len(zeros_sig)
    singleStageImpResponse = np.ones((numZeros, 2)).astype('complex64')
    singleStageImpResponse[:,1] = -1*zeros_sig
    ImpResponse = np.array([1])
    for ele in np.arange(numSamp-1):
        ImpResponse = np.convolve(singleStageImpResponse[ele,:], ImpResponse)

    return ImpResponse

def zeros2tf_fft(zeros_sig):
    """
    This function takes as input the signal zeros and outputs the FIR filter tap coefficients using the method of FFTs and IFFT
    inputs:
        1. zeros_sig: complex zeros of the signal

    outputs:
        1. ImpResponse: Impulse response obtained by successive convolution of (# zeros) of 2 tap FIR filters.
        The length of ImpResponse = # Zeros + 1

    Concept:
        An FIR system of N zeros can be modelled as N 2 tap cascaded FIR filters whose coefficients are h_i[n] = [1, -z_i]
        where h_i[n] is the impulse response of the ith filter and z_i is the ith zero of the system.
        Since the overall Z tranform is a product of the form (1-z_1*z^-1) * (1-z_2*z^-1) * ... ((1-z_N*z^-1)), the overall
        impulse response is given by h[n] = h1[n] conv h2[n] cov ....[hN[n]]
        The convolution is an expensive operation though. We could use the property that convolution in time domain
        is the same operation as multiplication in the freqeuncy domain. Also, FFTs and IFFTs are very optimized functions
        So we can perform the multiplication of the spectra of each impulse response
        and then take an IFFT to get back the time doamin final impulse response of the system.
        Note: We know that the final output should be of length N = # Zeros + 1. So we take an N point FFT of each of the 2 tap filters.
        We then multiply all the N-1(since N-1 zeros) N point FFTs to get a single N point FFT. We then take an N point IFFT to get the final system impulse response

        Hence, here in this function, we find the final imuplse response of the system by product of FFTs and then an IFFT
    """
    numZeros = len(zeros_sig)
    singleStageImpResponse = np.ones((numZeros, 2)).astype('complex64')
    singleStageImpResponse[:,1] = -1*zeros_sig
    FreqRespIndFilter = np.fft.fft(singleStageImpResponse,axis=1,n=numZeros+1)
    FreqRespFilter = np.prod(FreqRespIndFilter,axis=0)
    ImpResponse = np.fft.ifft(FreqRespFilter)

    return ImpResponse

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


def impulseRespGen(sigFreqInd_unipolar, interferFreqArray_bipolar, numOrigSamp, zerosFilterResponseSuperSet):

    """
    Inputs:
        1. sigFreqInd_unipolar: should be an integer index on the scale of numOrigSamp
        2. interferFreqArray_bipolar: should be an array on the scale of numOrigSamp
        Length of interferFreqArray_bipolar should be less than numZeros of system

    Outputs:
        1. signal_tf: Modified signal transfer function
        2. manipulatedZeros: Modified/manipulated zeros
    """

    respStartZeros = zerosFilterResponseSuperSet[:,sigFreqInd_unipolar]
    respStartPhases = np.angle(respStartZeros)

    interfererDigFreq = (2*np.pi/numOrigSamp)*(interferFreqArray_bipolar)
    interfererDigFreq_conj = -1*interfererDigFreq
    ind_closest_InterfererZeros, _ = min_angular_distance_index(interfererDigFreq_conj,respStartPhases)

    respPhaseManipulated = respStartPhases.copy()
    respPhaseManipulated[ind_closest_InterfererZeros] = interfererDigFreq_conj
    manipulatedZeros = np.exp(1j*respPhaseManipulated)

    # signal_tf, _ = sig.zpk2tf(manipulatedZeros,[],1)
    # signal_tf = zeros2tf_conv(manipulatedZeros)
    signal_tf = zeros2tf_fft(manipulatedZeros)

    """ The below energy normalization factor has been added so that the signal transfer function has unit energy.
    This is required because, the pseudo inverse implicitly has an AhA^-1 normalization implicitly.
    So the generated transfer function too (which mimics the pseudo inverse response)
    also needs to have an energy normalization."""
    signal_tf = signal_tf/(np.linalg.norm(signal_tf))**2

    return signal_tf, manipulatedZeros

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
