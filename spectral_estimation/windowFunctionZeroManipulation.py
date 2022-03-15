# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:53:53 2022

@author: saiguna
"""


import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

def angular_distance(theta_1, theta_2, mod=2*np.pi):

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

def filterResponse2zeros(filterResponseSuperSet):

    """
    This function takes in a matrix and generates the zeros of each column of the matrix.
    This function can be used to generate the zeros of a vandermonde matrix upfront and then store them
    inputs:
        1. filterResponseSuperSet: Matrix of sinusoids stacked as columns

    outputs:
        1. zerosFilterResponseSuperSet: Zeros of each column of filterResponseSuperSet stacked again as columns.
        Note: # rows of zerosFilterResponseSuperSet = # of rows of filterResponseSuperSet -1 (Since the number of zeros is 1 less than the filter tap length)
        This function uses the inbuilt scipy.signal.tf2zpk
    """

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
    ind_closest_InterfererZeros, _ = angular_distance(interfererDigFreq_conj,respStartPhases)

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


plt.close('all')

numOrigSamp = 168
numSamp = 14
numFFT = 1024


vandermondeMatrixSuperSet = generateVandermondeSuperSet(numOrigSamp, numSamp)
filterResponseSuperSet = np.conj(vandermondeMatrixSuperSet) # This has to have a conjugation effect to enhance the true tone
zerosFilterResponseSuperSet = filterResponse2zeros(filterResponseSuperSet)

signalbinsInteger_Bipolar = np.array([10,20]) # bin on a scale of numOrigSamp. Choose from [-numOrigSamp/2, numOrigSamp/2]
binIdx_unipolar = signalbinsInteger_Bipolar.copy()
binIdx_unipolar[binIdx_unipolar<0] += numOrigSamp

signalZeros = zerosFilterResponseSuperSet[:,binIdx_unipolar]

winfft = np.fft.fft(filterResponseSuperSet[:,binIdx_unipolar],n=numFFT, axis=0)
freAxis_bipolar = np.arange(-numFFT//2,numFFT//2)*2*np.pi/numFFT
winfftshift = np.fft.fftshift(winfft, axes=0)




""" Push the window zeros to the place of interferers"""
# Filter coefficients for signal 1
sigFreqInd_unipolar = binIdx_unipolar[0]
interferFreqArray_bipolar = np.array([signalbinsInteger_Bipolar[1]])
sig1_tf, zeros_sig1 = impulseRespGen(sigFreqInd_unipolar, interferFreqArray_bipolar, numOrigSamp, zerosFilterResponseSuperSet)

sig1_tf_fft = np.fft.fft(sig1_tf, n=numFFT)
sig1_tf_fft = np.fft.fftshift(sig1_tf_fft)


# Filter coefficients for signal 2
sigFreqInd_unipolar = binIdx_unipolar[1]
interferFreqArray_bipolar = np.array([signalbinsInteger_Bipolar[0]])
sig2_tf, zeros_sig2 = impulseRespGen(sigFreqInd_unipolar, interferFreqArray_bipolar, numOrigSamp, zerosFilterResponseSuperSet)

sig2_tf_fft = np.fft.fft(sig2_tf, n=numFFT)
sig2_tf_fft = np.fft.fftshift(sig2_tf_fft)

""" Frequency response of Vandermonde Pseudo inverse """
digFreq = signalbinsInteger_Bipolar*2*np.pi/numOrigSamp
phasor = np.exp(-1j*digFreq) # This negative is required since the pseudo inverse has a response conjugate to the true response

# vandermondeMatrix = np.exp(1j*digFreq[None,:]*np.arange(numSamp)[:,None])
vandermondeMatrix = vandermondeMatrixSuperSet[:,binIdx_unipolar]
vandermondeMatrixInv = np.linalg.pinv(vandermondeMatrix)
vandermondeMatrixInv_fft = np.fft.fft(vandermondeMatrixInv,axis=1, n=numFFT)
vandermondeMatrixInv_fft = np.fft.fftshift(vandermondeMatrixInv_fft,axes=1)

z_vandInv_row1, p_vanInv_row1, k = sig.tf2zpk(vandermondeMatrixInv[0,:],1)
z_vandInv_row2, p_vanInv_row2, k = sig.tf2zpk(vandermondeMatrixInv[1,:],1)

if 1:
    x = np.cos(np.linspace(0,2*np.pi,500))
    y = np.sin(np.linspace(0,2*np.pi,500))
    plt.figure(1,figsize=(20,10))
    plt.subplot(2,2,1)
    plt.title('Vandermonde Conjugate spectrum for signal 1')
    plt.plot(freAxis_bipolar, 20*np.log10(np.abs(winfftshift[:,0])));plt.grid(True)
    plt.xlabel('rad/samp')
    for i in range(numSamp-1):
        plt.axvline(np.angle(signalZeros[i,0]),c='m',lw=2,alpha=0.4)
    plt.grid(True)

    plt.subplot(2,2,2)
    plt.title('Zero plot for Vandermonde Conjugate for signal 1')
    plt.plot(signalZeros[:,0].real, signalZeros[:,0].imag, 'o', fillstyle='none', ms=14)
    plt.plot(x,y,color='k');
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('Real')
    plt.xlabel('Imag')

    plt.subplot(2,2,3)
    plt.title('Vandermonde Conjugate spectrum for signal 2')
    plt.plot(freAxis_bipolar, 20*np.log10(np.abs(winfftshift[:,1])));plt.grid(True)
    plt.xlabel('rad/samp')
    for i in range(numSamp-1):
        plt.axvline(np.angle(signalZeros[i,1]),c='m',lw=2,alpha=0.4)
    plt.grid(True)

    plt.subplot(2,2,4)
    plt.title('Zero plot for Vandermonde Conjugate for signal 2')
    plt.plot(signalZeros[:,1].real, signalZeros[:,1].imag, 'o', fillstyle='none', ms=14)
    plt.plot(x,y,color='k');
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('Real')
    plt.xlabel('Imag')


    plt.figure(2,figsize=(20,10))
    plt.suptitle('Frequency Response of Vandermonde pseudo inverse')
    plt.subplot(2,2,1)
    plt.title('Frequency Response of 1st row of pseudo inverse matrix')
    # plt.plot(freAxis_bipolar, 20*np.log10(np.abs(vandermondeMatrixInv_fft[0,:])/np.amax(np.abs(vandermondeMatrixInv_fft[0,:]))), color='k', label='Inv Vand Mat 1st row spectrum')
    plt.plot(freAxis_bipolar, 20*np.log10(np.abs(vandermondeMatrixInv_fft[0,:])), color='k', label='Inv Vand Mat 1st row spectrum')
    plt.axvline(-digFreq[0] ,color='b', label='signal')
    plt.axvline(-digFreq[1] , color='r', label='interferer')
    # plt.plot(freAxis_bipolar, 20*np.log10(np.abs(sig1_tf_fft)/np.amax(np.abs(sig1_tf_fft))), label='synthesized response for sig1')
    plt.plot(freAxis_bipolar, 20*np.log10(np.abs(sig1_tf_fft)), label='synthesized response for sig1')
    plt.xlabel('Dig Freq (rad/samp)')
    plt.grid(True)
    plt.legend()

    plt.subplot(2,2,2)
    plt.title('Zeros of 1st row of pseudo inverse matrix')
    plt.plot(z_vandInv_row1.real, z_vandInv_row1.imag, 'o', fillstyle='none', ms=14, label='Inv Vand Mat 1st row zeros')
    plt.plot(phasor[0].real, phasor[0].imag, 'd', fillstyle='none', ms=14, color='b', label='signal')
    plt.plot(phasor[1].real, phasor[1].imag, 's', fillstyle='none', ms=14, color='r', label='interferer')
    plt.plot(zeros_sig1.real, zeros_sig1.imag, '+', fillstyle='none', ms=14, color='k', label='zeros synthesized response for sig1')
    plt.plot(x,y,color='k');
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('Real')
    plt.xlabel('Imag')
    plt.legend()


    plt.subplot(2,2,3)
    plt.title('Frequency Response of 2nd row of pseudo inverse matrix')
    # plt.plot(freAxis_bipolar, 20*np.log10(np.abs(vandermondeMatrixInv_fft[1,:])/np.amax(np.abs(vandermondeMatrixInv_fft[1,:]))), color='k', label='Inv Vand Mat 1st row spectrum')
    plt.plot(freAxis_bipolar, 20*np.log10(np.abs(vandermondeMatrixInv_fft[1,:])), color='k', label='Inv Vand Mat 1st row spectrum')
    plt.axvline(-digFreq[1] , color='b', label='signal')
    plt.axvline(-digFreq[0] ,color='r', label='interferer')
    # plt.plot(freAxis_bipolar, 20*np.log10(np.abs(sig2_tf_fft)/(np.amax(np.abs(sig2_tf_fft)))), label='synthesized response for sig2')
    plt.plot(freAxis_bipolar, 20*np.log10(np.abs(sig2_tf_fft)), label='synthesized response for sig2')
    plt.xlabel('Dig Freq (rad/samp)')
    plt.grid(True)
    plt.legend()

    plt.subplot(2,2,4)
    plt.title('Zeros of 2nd row of pseudo inverse matrix')
    plt.plot(z_vandInv_row2.real, z_vandInv_row2.imag, 'o', fillstyle='none', ms=14, label='Inv Vand Mat 2nd row zeros')
    plt.plot(phasor[1].real, phasor[1].imag, 'd', fillstyle='none', ms=14, color='b', label='signal')
    plt.plot(phasor[0].real, phasor[0].imag, 's', fillstyle='none', ms=14, color='r', label='interferer')
    plt.plot(zeros_sig2.real, zeros_sig2.imag, '+', fillstyle='none', ms=14, color='k', label='zeros synthesized response for sig2')
    plt.plot(x,y,color='k');
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('Real')
    plt.xlabel('Imag')
    plt.legend()



    plt.figure(3,figsize=(20,10))
    plt.suptitle('Phase Response of Vandermonde pseudo inverse')
    plt.subplot(1,2,1)
    plt.title('Phase Response of 1st row of pseudo inverse matrix')
    # plt.plot(freAxis_bipolar, 20*np.log10(np.abs(vandermondeMatrixInv_fft[0,:])/np.amax(np.abs(vandermondeMatrixInv_fft[0,:]))), color='k', label='Inv Vand Mat 1st row spectrum')
    plt.plot(freAxis_bipolar, np.unwrap(np.angle(vandermondeMatrixInv_fft[0,:])), color='k', label='Inv Vand Mat 1st row spectrum')
    plt.axvline(-digFreq[0] ,color='b', label='signal')
    plt.axvline(-digFreq[1] , color='r', label='interferer')
    # plt.plot(freAxis_bipolar, 20*np.log10(np.abs(sig1_tf_fft)/np.amax(np.abs(sig1_tf_fft))), label='synthesized response for sig1')
    plt.plot(freAxis_bipolar, np.unwrap(np.angle(sig1_tf_fft)), lw=4, alpha=0.5, label='synthesized response for sig1')
    plt.xlabel('Dig Freq (rad/samp)')
    plt.grid(True)
    plt.legend()


    plt.subplot(1,2,2)
    plt.title('Phase Response of 2nd row of pseudo inverse matrix')
    # plt.plot(freAxis_bipolar, 20*np.log10(np.abs(vandermondeMatrixInv_fft[1,:])/np.amax(np.abs(vandermondeMatrixInv_fft[1,:]))), color='k', label='Inv Vand Mat 1st row spectrum')
    plt.plot(freAxis_bipolar, np.unwrap(np.angle(vandermondeMatrixInv_fft[1,:])), color='k', label='Inv Vand Mat 1st row spectrum')
    plt.axvline(-digFreq[1] , color='b', label='signal')
    plt.axvline(-digFreq[0] ,color='r', label='interferer')
    # plt.plot(freAxis_bipolar, 20*np.log10(np.abs(sig2_tf_fft)/(np.amax(np.abs(sig2_tf_fft)))), label='synthesized response for sig2')
    plt.plot(freAxis_bipolar, np.unwrap(np.angle(sig2_tf_fft)), lw=4, alpha=0.5, label='synthesized response for sig2')
    plt.xlabel('Dig Freq (rad/samp)')
    plt.grid(True)
    plt.legend()



