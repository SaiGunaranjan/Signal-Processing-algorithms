# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 13:19:32 2023

@author: Sai Gunaranjan
"""

"""
Scalloping loss is the loss in signal power when the frequency falls in the middle of 2 bins.
This is a parameter spec for every window function. The scalloping loss for SVA is same as that of rectangular window ~ 4 dB
More details about scalloping window are availabe here:
    https://www.recordingblogs.com/wiki/scalloping-loss#:~:text=The%20scalloping%20loss%20is%20the,the%20window%2C%20as%20explained%20below.
"""

import numpy as np
import matplotlib.pyplot as plt
from spectral_estimation_lib import spatially_variant_apodization_optimized

plt.close('all')

N = 512
binNum = 32
a1 = np.exp(1j*2*np.pi*binNum*np.arange(N)/N)
a2 = np.exp(1j*2*np.pi*(binNum+0.5)*np.arange(N)/N)


a1fft_rect = np.fft.fft(a1,n=N)/N
a2fft_rect = np.fft.fft(a2,n=N)/N
spectruma1_rect = 20*np.log10(np.abs(a1fft_rect))
spectruma2_rect = 20*np.log10(np.abs(a2fft_rect))
maxValRect = np.amax(spectruma1_rect)
spectruma1_rect = spectruma1_rect - maxValRect
spectruma2_rect = spectruma2_rect - maxValRect
scallopingLossRect = np.amax(spectruma2_rect)
print('Scalloping Loss for rectangular window = {0:.2f} dB'.format(scallopingLossRect))

a1fft_hann = np.fft.fft(a1*np.hanning(N),n=N)/N
a2fft_hann = np.fft.fft(a2*np.hanning(N),n=N)/N
spectruma1_hann = 20*np.log10(np.abs(a1fft_hann))
spectruma2_hann = 20*np.log10(np.abs(a2fft_hann))
maxValHann = np.amax(spectruma1_hann)
spectruma1_hann = spectruma1_hann - maxValHann
spectruma2_hann = spectruma2_hann - maxValHann
scallopingLossHann = np.amax(spectruma2_hann)
print('Scalloping Loss for Hanning window = {0:.2f} dB'.format(scallopingLossHann))


osrFact = 1
a1apod, _ = spatially_variant_apodization_optimized(a1, osrFact)
a1apod = np.fft.fftshift(a1apod)
a2apod, _ = spatially_variant_apodization_optimized(a2, osrFact)
a2apod = np.fft.fftshift(a2apod)
spectruma1_apod = 20*np.log10(np.abs(a1apod))
spectruma2_apod = 20*np.log10(np.abs(a2apod))
maxValApod = np.amax(spectruma1_apod)
spectruma1_apod = spectruma1_apod - maxValApod
spectruma2_apod = spectruma2_apod - maxValApod
scallopingLossApod = np.amax(spectruma2_apod)
print('Scalloping Loss for Apodization = {0:.2f} dB'.format(scallopingLossApod))

plt.figure(1,figsize=(20,10),dpi=150)
plt.suptitle('Scalloping loss (dB)')
plt.subplot(1,3,1)
plt.title('Rectangular window')
plt.plot(spectruma1_rect,label='signal on bin {}'.format(binNum))
plt.plot(spectruma2_rect,alpha=0.5, label='signal on bin {}'.format(binNum+0.5))
plt.grid(True)
plt.xlabel('bin Number')
plt.ylim([-10,3])
plt.xlim([binNum-3, binNum+3])
plt.legend()


plt.subplot(1,3,2)
plt.title('Hanning window')
plt.plot(spectruma1_hann, label='signal on bin {}'.format(binNum))
plt.plot(spectruma2_hann,alpha=0.5, label='signal on bin {}'.format(binNum+0.5))
plt.grid(True)
plt.xlabel('bin Number')
plt.ylim([-10,3])
plt.xlim([binNum-3, binNum+3])
plt.legend()

plt.subplot(1,3,3)
plt.title('Apodization')
plt.plot(spectruma1_apod, label='signal on bin {}'.format(binNum))
plt.plot(spectruma2_apod,alpha=0.5, label='signal on bin {}'.format(binNum+0.5))
plt.grid(True)
plt.xlabel('bin Number')
plt.ylim([-10,3])
plt.xlim([binNum-3, binNum+3])
plt.legend()
