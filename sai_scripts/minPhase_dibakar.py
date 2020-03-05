# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 19:23:02 2020

@author: Steradian Semi
"""

import numpy as np
import matplotlib.pyplot as plt

n = 32;
L = 16*n;
x = 2.5*np.random.randn(n) + 30;
x[0] = 100*np.sum(np.abs(x)); 
r = np.correlate(x,x,mode='full');
S =  np.abs(np.fft.fft(r,L));
ar = np.log(S)/2;
X = np.fft.fft(ar,L); 
X[0] = 0; 
X[1:(L//2)-1] = -1j*X[1:(L//2)-1]; 
X[L//2] = 0; 
X[L//2+1:-1] = 1j*X[L//2+1:-1];
ai = np.fft.ifft(X,L);
x_min = np.fft.ifft( np.exp(ar-1j*ai), L );
x_min = np.real(x_min[0:n])

plt.figure(1);
plt.subplot(1,2,1)
plt.plot(x,'-o',label='original min phase signal');
plt.plot(x_min,'-o',label='reconstructed min phase signal');
plt.legend();
plt.grid(True);
plt.subplot(1,2,2);
plt.title('Error b/w true and reconst')
plt.plot(x-x_min,'-o',label='true-recons');
#plt.ylim(-10000,10000)
plt.grid(True);
plt.legend()
