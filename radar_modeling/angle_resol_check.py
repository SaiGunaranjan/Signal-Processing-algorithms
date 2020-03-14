# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:42:22 2018

@author: lenovo
"""
import numpy as np
import matplotlib.pyplot as plt

Nrxe = 64
f0= 76.2e9
d=2200e-6
clight=3e8
Fs = clight/f0/d
raxis= np.arcsin((np.arange(Nrxe) - Nrxe/2)*Fs/Nrxe)*180/np.pi

data_store = 'D:\\Steradian\\Radar_Data\\recorded_data\\rangeBinCheck\\'
foldName = ['raw_data_0_15cm','raw_data_1_20cm','raw_data_2_24cm','raw_data_3_30cm','raw_data_4_35_36cm','raw_data_5_42cm','raw_data_6_50cm']
N= len(foldName)
Q1_fft_db = np.zeros((N,1024,8,Nrxe))
Q2_fft_db = np.zeros((N,1024,8,Nrxe))
Q0_fft_db = np.zeros((N,1024,Nrxe))
for i in range(N):
    P = np.load(data_store+foldName[i]+'\\'+'Radar_Data_299.npy')
    Q = P.reshape(-1,18,4)
    Q1a = Q[:,0:8,:]
    Q2a = Q[:,8:16,:]
    Q1 = np.pad(Q1a,((0,0),(0,0),(0,60)), 'constant') # zeropad the doppler
    Q2 = np.pad(Q2a,((0,0),(0,0),(0,60)), 'constant') # zeropad the sensor axis
    Q1_fft = np.fft.fftshift(np.fft.fft(np.fft.fft(Q1,axis=1),axis=2),axes=2)
    Q2_fft = np.fft.fftshift(np.fft.fft(np.fft.fft(Q2,axis=1),axis=2),axes=2)
    Q1_fft_db[i,:,:,:] = 20*np.log10(np.abs(Q1_fft)) -20
    Q2_fft_db[i,:,:,:] = 20*np.log10(np.abs(Q2_fft))
    
    Q1 = np.pad(Q1a,((0,0),(0,0),(0,0)), 'constant') # zeropad the sensor axis
    Q2 = np.pad(Q2a,((0,0),(0,0),(0,0)), 'constant') # zeropad the sensor axis
    Q1_fft = np.fft.fft(Q1,axis=1)
    Q2_fft = np.fft.fft(Q2,axis=1)
    Q0_fft = np.zeros((Q1.shape[0],7)).astype('complex64')
    Q0_fft[:,0:3] = Q1_fft[:,0,0:3]
    Q0_fft[:,4:7] = Q2_fft[:,0,1:4]
    Q0_fft[:,3] = (Q1_fft[:,0,3] + Q2_fft[:,0,0]) /2
    Q0 = np.pad(Q0_fft,((0,0),(0,Nrxe-7)), 'constant') # zeropad the doppler
    Q0_fft = np.fft.fftshift(np.fft.fft(Q0,axis=1),axes=1)
    Q0_fft_db[i,:,:] = 20*np.log10(np.abs(Q0_fft)) -20
    
plt.figure(1)    
plt.plot(raxis,Q1_fft_db[:,13,0,:].T,lw=2,alpha=0.8)
plt.legend(foldName)
#plt.figure(2)    
#plt.plot(Q1_fft_db[:,14,0,:].T,lw=2,alpha=0.8)
#plt.legend(foldName)
plt.figure(3)    
plt.plot(raxis,Q0_fft_db[:,13,:].T,lw=2,alpha=0.8)
plt.legend(foldName)
