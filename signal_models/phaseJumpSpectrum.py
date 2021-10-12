# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 19:50:05 2021

@author: Sai Gunaranjan Pelluri
"""

""" This script models the impact of a phase jump on a linear phase signal on the Spectrum of the signal.
A tiny phase jump manifests as tiny kink/side lobe in the magnitude spectrum thus reducing the dynamic range of the
spectrum"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

t1 = np.exp(1j*np.arange(74)*140/74)
t2 = np.hstack((np.zeros(44),2*np.ones(30)))
t3 = np.exp(1j*t2) * t1
plt.figure(4, figsize=(20,10));
plt.suptitle('Phase jump of 2 rad introduced at the 45th sample')
plt.subplot(1,2,1)
plt.title('unwrapped phase (rad)')
plt.plot(np.unwrap(np.angle(t3)),'-o')
plt.xlabel('ULA channel number')
plt.grid(True)
t4 = t3*np.hanning(t1.shape[0])
t5 = np.fft.fftshift(np.fft.fft(t4,n=2048))
t6 = 20*np.log10(np.abs(t5))
t7 = t6-np.amax(t6)
plt.subplot(1,2,2)
plt.title('Spectrum')
plt.plot(t7)
plt.xlabel('FFT bins')
plt.grid(True)


t1 = np.exp(1j*np.arange(74)*140/74)
t2 = np.hstack((np.zeros(44),-2*np.ones(30)))
t3 = np.exp(1j*t2) * t1
plt.figure(5, figsize=(20,10));
plt.suptitle('Phase jump of -2 rad introduced at the 45th sample')
plt.subplot(1,2,1)
plt.title('unwrapped phase (rad)')
plt.plot(np.unwrap(np.angle(t3)),'-o')
plt.xlabel('ULA channel number')
plt.grid(True)
t4 = t3*np.hanning(t1.shape[0])
t5 = np.fft.fftshift(np.fft.fft(t4,n=2048))
t6 = 20*np.log10(np.abs(t5))
t7 = t6-np.amax(t6)
plt.subplot(1,2,2)
plt.title('Spectrum')
plt.plot(t7)
plt.xlabel('FFT bins')
plt.grid(True)

t1 = np.exp(1j*np.arange(74)*140/74)
t2 = np.hstack((np.zeros(44),np.pi*np.ones(30)))
t3 = np.exp(1j*t2) * t1
plt.figure(6,figsize=(20,10));
plt.suptitle('Phase jump of pi rad introduced at the 45th sample')
plt.subplot(1,2,1)
plt.title('unwrapped phase (rad)')
plt.plot(np.unwrap(np.angle(t3)),'-o')
plt.xlabel('ULA channel number')
plt.grid(True)
t4 = t3*np.hanning(t1.shape[0])
t5 = np.fft.fftshift(np.fft.fft(t4,n=2048))
t6 = 20*np.log10(np.abs(t5))
t7 = t6-np.amax(t6)
plt.subplot(1,2,2)
plt.title('Spectrum')
plt.plot(t7)
plt.xlabel('FFT bins')
plt.grid(True)