# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 18:30:18 2023

@author: Sai Gunaranjan
"""

"""

Spatially Variant Apodization

In this script, I have implemented a technique called Spatially Variant Apodization (SVA). SVA is a like for like replacement
for a windowed FFT. A typical windowed FFT is characterized by main lobe width and side lobe level.
Rectangular window has very good main lobe width but poor SLLs (13 dBc). Hanning window has poor main lobe width but good side
SLLs (31 dBc). SVA is a technique which combines the best of both these windows. It gives the main lobe width of a rectangular window
while giving the SLL of a Hanning window. Thus it gives the best of both worlds. The working principle of SVA is as follows:

The SVA attemps to select a different window function for each frequency. Let us understand this more closely.
A raised cosine window is a generic window which is controlled by an alpha parameter. The functional form of the window is
1-2*alpha*cos(2*pi/N*n), where 'alpha' lies in [0,0.5], 'N' is the number of samples and 'n' varies from 0 to N-1.
When 'alpha' = 0, the raised cosine window becomes a rectangular window. When 'alpha' = 0.5, the window becomes a hanning window.
By varying alpha from 0 to 0.5 in fine steps, we can create multiple variants of the raised cosine window.
A brute force version of the SVA is as follows. Subject the signal to each of these windows and perform an FFT on each of
the windowed signals. This results in several FFTs of the signal each with a different window. Now, compute the psd/magnitude square
of each of these FFTs. To obtain the final psd/spectrum, take minimum along the window axis. This results in a spectrum which has
minimum energy at each frequency point across all window function. In essence, we are selecting a window function which generates
the minimum energy at each frequency point. Since, we are selecting munimum energy at each frequency point (across all windows),
this will result in minimum SLLs as well as best main lobe width. But this brute force method requires heavy compute since
we need to generate several windows and multiply the signal also with each of these windows and compute FFT on each of these signals.
Finally we also need to take a minimum for each of the frequency point. This is serioulsy heavy compute as well as memory.
This can be circumvented by implementing the optimized version of the SVA. For this, we need to take a closer look at the structure
of the raised cosine window. The functional form of the window is w[n] = 1-2*alpha*cos(2*pi/N*n).
If we assume the signal of interest is x[n], then the windowed signal is y[n] = x[n]w[n]. Now multiplication in time domain is
equivalent to convolution in the frequency domain. So if we were to analyse the windowed signal in the Fourier domain,
we get Y(w) = X(w) * W(w), where * denotes convolution operation. Now W(w) = 2pi * ( delta(w) - alpha*[delta(w-2*pi/N) + delta(w+2*pi/N)]).
So,  Y(w) = 2pi * (X(w) - alpha*[X(w-2pi/N) + X(w+2pi/N)]). Now, the optimum value of alpha(at each frequency) is chosen by
minimizing the magnitude squared of Y(w) over all alpha lying between 0 to 0.5. This is a constrained optimization problem.
Solving this, we obtain a closed form expression for alpha at each frequency point omega.
Hence alpha is a function of the omega chosen. The other details of the derivation are available in the paper link below:
https://saigunaranjan.atlassian.net/wiki/spaces/RM/pages/22249477/Spatially+Variant+Apodization

After obtaining a closed form expression for alpha and substituting in the expression for Y(w), we obtain the spectrum for SVA.
This method of computing the SVA spectrum is computationally light. As evident from the closed form expression for alpha,
the only major compute (apart from the FFT computation of x[n]) is an N point division for an N point oversampled FFT.
So there are N complex divisions corresponding to each of the N frequency points.
I have implemented both the brute force method as well as the optimized method of SVA and have generated the results.
Both thse methods seem to be closely matching. The results are on expected lines with the SVA offering the best SLLS as well as
main lobe width.

Currently I have tested and observed the magnitude spectrum performance of SVA. Need to check if SVA can also be used to extract
the phase information from the complex spectrum. In other words, is SVA (optimized version) a spectral estimator like APES or
a pseudo spectral estimator like MUSIC which gives info about the sinusoids present in the signal but not the phase of the sinusoids.
One thing is clear, the SVA-brute force is a pseudo spectral estimator since we pick the maximum energy across several windows
for each frequency point. SInce it operates on the energy to obatin the final spectrum, it is a pseudo spectral estimator. But I need
to check if SVA-optimized is a spectral estimator which gives the phase of the sinusoid as well.

"""

import numpy as np
import matplotlib.pyplot as plt
from spectral_estimation_lib import spatially_variant_apodization_optimized
from scipy.signal import argrelextrema

# np.random.seed(30)

plt.close('all')

num_sources = 1

c = 3e8
fc = 79e9
lamda = c/fc
mimoSpacing = lamda/2
fsSpatial = lamda/mimoSpacing


osrFact = 2#16 #128
num_samples = 1024#32
numFFTOSR = osrFact*num_samples


## RF parameters
thermalNoise = -174 # dBm/Hz
noiseFigure = 10 # dB
baseBandgain = 34 #dB
adcSamplingRate = 56.25e6 # 56.25 MHz
adcSamplingTime = 1/adcSamplingRate # s
dBFs_to_dBm = 10
totalNoisePower_dBm = thermalNoise + noiseFigure + baseBandgain + 10*np.log10(adcSamplingRate)
totalNoisePower_dBFs = totalNoisePower_dBm - 10
noiseFloor_perBin = -90#totalNoisePower_dBFs - 10*np.log10(num_samples) # dBFs/bin
noisePower_perBin = 10**(noiseFloor_perBin/10)
totalNoisePower = noisePower_perBin*num_samples # sigmasquare totalNoisePower
noise_sigma = np.sqrt(totalNoisePower)

snrArr = np.arange(10,150,10)#np.arange(10,90,20)
numSNR = len(snrArr)
numMC = 100

angleSLLMatrixRect_percentile = np.zeros((numSNR))
angleSLLMatrixHann_percentile = np.zeros((numSNR))
angleSLLMatrixSVA_percentile = np.zeros((numSNR))

percentile = 99
count_snrMC = 0
for object_snr in snrArr:
    signalPowerdBFs = noiseFloor_perBin + object_snr
    signalPower = 10**(signalPowerdBFs/10)
    signalAmplitude = np.sqrt(signalPower)
    signalPhase = np.exp(1j*np.random.uniform(-np.pi, np.pi,size = num_sources))
    complex_signal_amplitudes = signalAmplitude*signalPhase

    random_freq = np.random.uniform(low=-np.pi, high=np.pi, size = 1)
    source_freq = np.array([random_freq])
    source_angle_deg = np.arcsin((source_freq/(2*np.pi))*fsSpatial)*180/np.pi

    source_signals = np.matmul(np.exp(-1j*np.outer(np.arange(num_samples),source_freq)),complex_signal_amplitudes[:,None])
    angleSllArrayRect = np.empty([0])
    angleSllArrayHann = np.empty([0])
    angleSllArraySVA = np.empty([0])
    for mcarlo in range(numMC):
        wgn_noise = np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape)
        received_signal = source_signals + wgn_noise

        magnitude_spectrum_fft = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(received_signal,axis=0, n=numFFTOSR)/received_signal.shape[0],axes=0)))
        magnitude_spectrum_fft -= np.amax(magnitude_spectrum_fft,axis=0)[None,:]
        magnitude_spectrum_fft = magnitude_spectrum_fft[:,0]

        magnitude_spectrum_fft_hann = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(received_signal*np.hanning(num_samples)[:,None],axis=0, n=numFFTOSR)/received_signal.shape[0],axes=0)))
        magnitude_spectrum_fft_hann -= np.amax(magnitude_spectrum_fft_hann,axis=0)[None,:]
        magnitude_spectrum_fft_hann = magnitude_spectrum_fft_hann[:,0]

        """ SVA Optimized"""
        svaOptimalComplexSpectrumfftshifted, svaOptimalMagSpectrumdB = spatially_variant_apodization_optimized(received_signal, osrFact)


        """ SLL computation Rect"""
        localMaxInd = argrelextrema(magnitude_spectrum_fft,np.greater_equal,axis=0,order=1)[0]
        try:
            sllInd = np.argsort(magnitude_spectrum_fft[localMaxInd])[-2] # 1st SLL
            sllValdBcRect = magnitude_spectrum_fft[localMaxInd[sllInd]]
        except IndexError:
            sllValdBcRect = 0
        angleSllArrayRect = np.hstack((angleSllArrayRect,sllValdBcRect))

        """ SLL computation Hann"""
        localMaxInd = argrelextrema(magnitude_spectrum_fft_hann,np.greater_equal,axis=0,order=1)[0]
        try:
            sllInd = np.argsort(magnitude_spectrum_fft_hann[localMaxInd])[-2] # 1st SLL
            sllValdBcHann = magnitude_spectrum_fft_hann[localMaxInd[sllInd]]
        except IndexError:
            sllValdBcHann = 0
        angleSllArrayHann = np.hstack((angleSllArrayHann,sllValdBcHann))

        """ SLL computation SVA"""
        localMaxInd = argrelextrema(svaOptimalMagSpectrumdB,np.greater_equal,axis=0,order=1)[0]
        try:
            sllInd = np.argsort(svaOptimalMagSpectrumdB[localMaxInd])[-2] # 1st SLL
            sllValdBcSVA = svaOptimalMagSpectrumdB[localMaxInd[sllInd]]
        except IndexError:
            sllValdBcSVA = 0
        angleSllArraySVA = np.hstack((angleSllArraySVA,sllValdBcSVA))



    angleSLLMatrixRect_percentile[count_snrMC] = np.percentile(angleSllArrayRect,percentile)
    angleSLLMatrixHann_percentile[count_snrMC] = np.percentile(angleSllArrayHann,percentile)
    angleSLLMatrixSVA_percentile[count_snrMC] = np.percentile(angleSllArraySVA,percentile)

    count_snrMC += 1


n = 0
plt.figure(n+1,figsize=(20,10), dpi=200)
plt.title('{} ile Angle SLLs(dBc) vs SNR'.format(percentile))
plt.plot(snrArr, angleSLLMatrixRect_percentile, '-o',label='Rect Win')
plt.plot(snrArr, angleSLLMatrixHann_percentile, '-o',label='Hann Win')
plt.plot(snrArr, angleSLLMatrixSVA_percentile, '-o',label= 'SVA')
# plt.axhline(WindSLL,color='k',label='Hanning Window SLL',linestyle='dashed')
plt.xlabel('SNR (dB)')
plt.ylabel('SLL (dBc)')
plt.grid(True)
plt.legend()
plt.ylim([-80,10])
n+=1















