# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:53:53 2022

@author: saiguna
"""


import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

def angular_distance(theta_1, theta_2, mod=2*np.pi):
    difference = np.abs(theta_1[:,None] % mod - theta_2[None,:] % mod)
    return np.minimum(difference, mod - difference)

def zeros2tf_conv(zeros_sig1):
    numZeros = len(zeros_sig1)
    singleStageImpResponse = np.ones((numZeros, 2)).astype('complex64')
    singleStageImpResponse[:,1] = -1*zeros_sig1
    ImpResponse = np.array([1])
    for ele in np.arange(numSamp-1):
        ImpResponse = np.convolve(singleStageImpResponse[ele,:], ImpResponse)

    return ImpResponse

def zeros2tf_fft(zeros_sig1):
    numZeros = len(zeros_sig1)
    singleStageImpResponse = np.ones((numZeros, 2)).astype('complex64')
    singleStageImpResponse[:,1] = -1*zeros_sig1
    FreqRespIndFilter = np.fft.fft(singleStageImpResponse,axis=1,n=numZeros+1)
    FreqRespFilter = np.prod(FreqRespIndFilter,axis=0)
    ImpResponse = np.fft.ifft(FreqRespFilter)

    return ImpResponse

def filterResponse2zeros(filterResponseSuperSet):

    numResponses = filterResponseSuperSet.shape[1]
    numZeros = filterResponseSuperSet.shape[0] - 1
    zerosFilterResponseSuperSet = np.zeros((numZeros,numResponses),dtype=np.complex64)
    for ele in np.arange(numResponses):
        zerosFilterResponseSuperSet[:,ele], _, _ = sig.tf2zpk(filterResponseSuperSet[:,ele], 1)

    return zerosFilterResponseSuperSet



plt.close('all')

numOrigSamp = 168
numSamp = 14
numFFT = 1024


vandermondeMatrixSuperSet = np.exp(1j*2*np.pi/numOrigSamp*(np.arange(numSamp)[:,None]*np.arange(-numOrigSamp//2, numOrigSamp//2)[None,:]))
vandermondeMatrixSuperSet = np.fft.fftshift(vandermondeMatrixSuperSet, axes=1)
filterResponseSuperSet = np.conj(vandermondeMatrixSuperSet) # This has to have a conjugation effect to enhancing the true tone
zerosFilterResponseSuperSet = filterResponse2zeros(filterResponseSuperSet)

signalbinsInteger_Bipolar = np.array([10,20]) # bin on a scale of numOrigSamp. Choose from [-numOrigSamp/2, numOrigSamp/2]
binIdx_unipolar = signalbinsInteger_Bipolar.copy()
binIdx_unipolar[binIdx_unipolar<0] += numOrigSamp
signalZeros = zerosFilterResponseSuperSet[:,binIdx_unipolar]


winfft = np.fft.fft(filterResponseSuperSet[:,binIdx_unipolar],n=numFFT, axis=0)
freAxis_bipolar = np.arange(-numFFT//2,numFFT//2)*2*np.pi/numFFT
winfftshift = np.fft.fftshift(winfft, axes=0)


x = np.cos(np.linspace(0,2*np.pi,500))
y = np.sin(np.linspace(0,2*np.pi,500))
plt.figure(1,figsize=(20,10))
plt.subplot(2,2,1)
plt.title('Vandermonde Conjugate spectrum')
plt.plot(freAxis_bipolar, 20*np.log10(np.abs(winfftshift[:,0])));plt.grid(True)
plt.xlabel('rad/samp')
for i in range(numSamp-1):
    plt.axvline(np.angle(signalZeros[i,0]),c='m',lw=2,alpha=0.4)
plt.grid(True)


plt.subplot(2,2,2)
plt.title('Zero plot for Kaiser window')
plt.plot(signalZeros[:,0].real, signalZeros[:,0].imag, 'o', fillstyle='none', ms=14)
plt.plot(x,y,color='k');
plt.axis('equal')
plt.grid(True)
plt.xlabel('Real')
plt.xlabel('Imag')

plt.subplot(2,2,3)
plt.title('Vandermonde Conjugate spectrum')
plt.plot(freAxis_bipolar, 20*np.log10(np.abs(winfftshift[:,1])));plt.grid(True)
plt.xlabel('rad/samp')
for i in range(numSamp-1):
    plt.axvline(np.angle(signalZeros[i,1]),c='m',lw=2,alpha=0.4)
plt.grid(True)


plt.subplot(2,2,4)
plt.title('Zero plot for Kaiser window')
plt.plot(signalZeros[:,1].real, signalZeros[:,1].imag, 'o', fillstyle='none', ms=14)
plt.plot(x,y,color='k');
plt.axis('equal')
plt.grid(True)
plt.xlabel('Real')
plt.xlabel('Imag')





bin1 = signalbinsInteger_Bipolar[0] + np.random.uniform(-0.5,0.5,1)[0] # bin on a scale of numOrigSamp. Choose from [-numOrigSamp/2, numOrigSamp/2]
bin2 = signalbinsInteger_Bipolar[1] + np.random.uniform(-0.5,0.5,1)[0]
bins = np.array([bin1,bin2])
digFreq = bins*2*np.pi/numOrigSamp
phasor = np.exp(-1j*digFreq) # This negative is required since the pseudo inverse has a response conjugate to the true response



""" Push the window zeros to the place of interferers"""
# Filter coefficients for signal 1
phase_window = np.angle(signalZeros[:,0])
signalFreq = digFreq[0]
interfererFreq = -1*np.array([digFreq[1]])
ind_closest_winZeros = np.argmin(angular_distance(interfererFreq,phase_window),axis=1)

phase_window_manipulated = phase_window.copy()
phase_window_manipulated[ind_closest_winZeros] = interfererFreq
zeros_sig1 = np.exp(1j*phase_window_manipulated)

# sig1_tf, _ = sig.zpk2tf(zeros_sig1,[],1)
# sig1_tf = zeros2tf_conv(zeros_sig1)
sig1_tf = zeros2tf_fft(zeros_sig1)

sig1_tf_fft = np.fft.fft(sig1_tf, n=numFFT)
sig1_tf_fft = np.fft.fftshift(sig1_tf_fft)


# Filter coefficients for signal 2
phase_window = np.angle(signalZeros[:,1])
signalFreq = digFreq[1]
interfererFreq = -1*np.array([digFreq[0]]) # Need to place the zeros at the negative of the interferer locations.
ind_closest_winZeros = np.argmin(angular_distance(interfererFreq,phase_window),axis=1)

phase_window_manipulated = phase_window.copy()
phase_window_manipulated[ind_closest_winZeros] = interfererFreq
zeros_sig2 = np.exp(1j*phase_window_manipulated)

# sig2_tf, _ = sig.zpk2tf(zeros_sig2,[],1)
# sig2_tf = zeros2tf_conv(zeros_sig2)
sig2_tf = zeros2tf_fft(zeros_sig2)

sig2_tf_fft = np.fft.fft(sig2_tf, n=numFFT)
sig2_tf_fft = np.fft.fftshift(sig2_tf_fft)

vandermondeMatrix = np.exp(1j*digFreq[None,:]*np.arange(numSamp)[:,None])
vandermondeMatrixInv = np.linalg.pinv(vandermondeMatrix)
vandermondeMatrixInv_fft = np.fft.fft(vandermondeMatrixInv,axis=1, n=numFFT)
vandermondeMatrixInv_fft = np.fft.fftshift(vandermondeMatrixInv_fft,axes=1)

z_vandInv_row1, p_vanInv_row1, k = sig.tf2zpk(vandermondeMatrixInv[0,:],1)
z_vandInv_row2, p_vanInv_row2, k = sig.tf2zpk(vandermondeMatrixInv[1,:],1)


plt.figure(2,figsize=(20,10))
plt.suptitle('Frequency Response of Vandermonde pseudo inverse')
plt.subplot(2,2,1)
plt.title('Frequency Response of 1st row of pseudo inverse matrix')
plt.plot(freAxis_bipolar, 20*np.log10(np.abs(vandermondeMatrixInv_fft[0,:])/np.amax(np.abs(vandermondeMatrixInv_fft[0,:]))), color='k', label='Inv Vand Mat 1st row spectrum')
plt.axvline(-digFreq[0] ,color='b', label='signal')
plt.axvline(-digFreq[1] , color='r', label='interferer')
plt.plot(freAxis_bipolar, 20*np.log10(np.abs(sig1_tf_fft)/np.amax(np.abs(sig1_tf_fft))), label='synthesized response for sig1')
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
plt.plot(freAxis_bipolar, 20*np.log10(np.abs(vandermondeMatrixInv_fft[1,:])/np.amax(np.abs(vandermondeMatrixInv_fft[1,:]))), color='k', label='Inv Vand Mat 1st row spectrum')
plt.axvline(-digFreq[1] , color='b', label='signal')
plt.axvline(-digFreq[0] ,color='r', label='interferer')
plt.plot(freAxis_bipolar, 20*np.log10(np.abs(sig2_tf_fft)/(np.amax(np.abs(sig2_tf_fft)))), label='synthesized response for sig2')
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
