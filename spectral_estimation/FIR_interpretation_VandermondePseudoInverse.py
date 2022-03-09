# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 13:54:47 2022

@author: saiguna
"""

""" This script analyzes the spectral response of Least Squares based pseudo inverse for a Vandermonde matrix.
It turns out that the pseudo inverse based coefficents(for each row of the vandermonde pseudo inverse) can be modelled
as a simple FIR filter with number of taps equal to the number of columns of the vandermonde pseudo inverse matrix"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

numOrigSamp = 168
numSamp = 14
numFFT = 1024

x = np.cos(np.linspace(0,2*np.pi,500))
y = np.sin(np.linspace(0,2*np.pi,500))

bin1 = 10 + np.random.uniform(-0.5,0.5,1)[0] # bin on a scale of numOrigSamp
bin2 = 13 + np.random.uniform(-0.5,0.5,1)[0]

bins = np.array([bin1,bin2])
digFreq = bins*2*np.pi/numOrigSamp
vandermondeMatrix = np.exp(1j*digFreq[None,:]*np.arange(numSamp)[:,None])

phasor = np.exp(-1j*digFreq)

vandermondeMatrixInv = np.linalg.pinv(vandermondeMatrix)
vandermondeMatrixInv_fft = np.fft.fft(vandermondeMatrixInv,axis=1, n=numFFT)

z_vandInv_row1, p_vanInv_row1, k = sig.tf2zpk(vandermondeMatrixInv[0,:],1)
z_vandInv_row2, p_vanInv_row2, k = sig.tf2zpk(vandermondeMatrixInv[1,:],1)

freAxis_unipolar = np.arange(0,numFFT)*2*np.pi/numFFT

plt.figure(2,figsize=(20,10))
plt.suptitle('Frequency Response of Vandermonde pseudo inverse')
plt.subplot(2,2,1)
plt.title('Frequency Response of 1st row of pseudo inverse matrix')
plt.plot(freAxis_unipolar, 20*np.log10(np.abs(vandermondeMatrixInv_fft[0,:])), label='Inv Vand Mat 1st row spectrum')
plt.axvline(-digFreq[0] + 2*np.pi,color='orange', label='signal')
plt.axvline(-digFreq[1] + 2*np.pi, color='m', label='interferer')
plt.xlabel('Dig Freq (rad/samp)')
plt.grid(True)
plt.legend()

plt.subplot(2,2,2)
plt.title('Zeros of 1st row of pseudo inverse matrix')
plt.plot(z_vandInv_row1.real, z_vandInv_row1.imag, 'o', fillstyle='none', ms=14, label='Inv Vand Mat 1st row zeros')
plt.plot(phasor[0].real, phasor[0].imag, 'd', fillstyle='none', ms=14, color='orange', label='signal')
plt.plot(phasor[1].real, phasor[1].imag, 's', fillstyle='none', ms=14, color='m', label='interferer')
plt.plot(x,y,color='k');
plt.axis('equal')
plt.grid(True)
plt.xlabel('Real')
plt.xlabel('Imag')
plt.legend()


plt.subplot(2,2,3)
plt.title('Frequency Response of 2nd row pseudo inverse matrix')
plt.plot(freAxis_unipolar, 20*np.log10(np.abs(vandermondeMatrixInv_fft[1,:])), label='Inv Vand Mat 1st row spectrum')
plt.axvline(-digFreq[1] + 2*np.pi, color='orange', label='signal')
plt.axvline(-digFreq[0] + 2*np.pi,color='magenta', label='interferer')
plt.xlabel('Dig Freq (rad/samp)')
plt.grid(True)
plt.legend()

plt.subplot(2,2,4)
plt.title('Zeros of 2nd row of pseudo inverse matrix')
plt.plot(z_vandInv_row2.real, z_vandInv_row2.imag, 'o', fillstyle='none', ms=14, label='Inv Vand Mat 2nd row zeros')
plt.plot(phasor[1].real, phasor[1].imag, 'd', fillstyle='none', ms=14, color='orange', label='signal')
plt.plot(phasor[0].real, phasor[0].imag, 's', fillstyle='none', ms=14, color='m', label='interferer')
plt.plot(x,y,color='k');
plt.axis('equal')
plt.grid(True)
plt.xlabel('Real')
plt.xlabel('Imag')
plt.legend()
