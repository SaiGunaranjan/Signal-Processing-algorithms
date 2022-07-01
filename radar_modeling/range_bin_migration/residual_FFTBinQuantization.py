# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:33:44 2022

@author: saiguna
"""

""" This script captures the residual magnitude and phase due to FFT bin quantization

The residual on a particular bin is given by
Sum (exp(1j*(2pi/N)*alpha*n)) = exp(1j*pi*(1-1/N))*sin(pi*alpha)/sin(pi*alpha/N)

The link to the derivation is available in the below confluence page link:
    https://saigunaranjan.atlassian.net/wiki/spaces/RM/pages/3244051/Residual+due+to+FFT+bin+Quantization

"""


import numpy as np
import matplotlib.pyplot as plt


alpha = np.linspace(0,1,100)
N = 140

fun = np.zeros(alpha.shape)
num = np.sin(np.pi*alpha)
den = np.sin(np.pi*alpha/N)
fun[1::] = num[1::]/den[1::]
fun[0] = N

phasor = np.exp(1j*(np.pi*alpha*(1-1/N)))

residual = phasor*fun

plt.figure(1,dpi=200,figsize=(20,10))
plt.subplot(1,2,1)
plt.title('Residual Magnitude(linear) on a quantized bin')
plt.plot(alpha, np.abs(residual))
plt.xlabel('alpha')
plt.grid(True)

plt.subplot(1,2,2)
plt.title('Residual Phase on a quantized bin')
plt.plot(alpha, np.unwrap(np.angle(residual)))
plt.axhline(np.pi-np.pi/N, color='k', linestyle = '--',label='PI - PI/N')
plt.xlabel('alpha')
plt.ylabel('Phase (rad)')
plt.legend()
plt.grid(True)