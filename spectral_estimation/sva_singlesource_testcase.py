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
"""

import numpy as np
import matplotlib.pyplot as plt
import spectral_estimation_lib as spec_est

np.random.seed(30)

plt.close('all')
num_samples = 32
c = 3e8
fc = 79e9
lamda = c/fc
mimoSpacing = lamda/2
fsSpatial = lamda/mimoSpacing
nativeAngResDeg = np.arcsin(fsSpatial/num_samples)*180/np.pi
# print('Native Angular Resolution = {0:.2f} deg'.format(nativeAngResDeg))
num_sources = 1
object_snr = np.array([40])
noise_power_db = -40 # Noise Power in dB
noise_variance = 10**(noise_power_db/10)
noise_sigma = np.sqrt(noise_variance)
weights = noise_variance*(10**(object_snr/10))
signal_phases = np.exp(1j*np.random.uniform(low=-np.pi, high = np.pi, size = num_sources))
complex_signal_amplitudes = weights*signal_phases
random_freq = np.random.uniform(low=-np.pi, high=np.pi, size = 1)
fft_resol_fact = 2
resol_fact = 0.65
digFreqRes = resol_fact*2*np.pi/num_samples
angResDeg = np.arcsin((digFreqRes/(2*np.pi))*fsSpatial)*180/np.pi
# print('Programmed Angular Resolution = {0:.2f} deg'.format(angResDeg))
source_freq = np.array([random_freq])
source_angle_deg = np.arcsin((source_freq/(2*np.pi))*fsSpatial)*180/np.pi
spectrumGridOSRFact = 128 # 32 if we want a less oversampled spectrum evaluation
digital_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/(spectrumGridOSRFact*num_samples))
angleGrid = np.arcsin(((digital_freq_grid/(2*np.pi))*fsSpatial))*180/np.pi
source_signals = np.matmul(np.exp(-1j*np.outer(np.arange(num_samples),source_freq)),complex_signal_amplitudes[:,None])
wgn_noise = np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape)
received_signal = source_signals + wgn_noise
corr_mat_model_order = num_samples//2-2 # must be strictly less than num_samples/2

magnitude_spectrum_fft = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(received_signal,axis=0, n=len(digital_freq_grid))/received_signal.shape[0],axes=0)))
phase_spectrum_fft = np.unwrap(np.angle(np.fft.fftshift(np.fft.fft(received_signal,axis=0, n=len(digital_freq_grid))/received_signal.shape[0],axes=0)))
magnitude_spectrum_fft -= np.amax(magnitude_spectrum_fft,axis=0)[None,:]

magnitude_spectrum_fft_hann = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(received_signal*np.hanning(num_samples)[:,None],axis=0, n=len(digital_freq_grid))/received_signal.shape[0],axes=0)))
magnitude_spectrum_fft_hann -= np.amax(magnitude_spectrum_fft_hann,axis=0)[None,:]


""" SVA brute force"""
alpha = np.linspace(0,0.5,100)
svaWindow = 1 - 2*alpha[None,:]*np.cos(2*np.pi*np.arange(num_samples)[:,None]/num_samples)
signalWindowed = received_signal * svaWindow
svaSignalFFT = np.fft.fft(signalWindowed,n=len(digital_freq_grid),axis=0)/num_samples
svaSignalFFTShift = np.fft.fftshift(svaSignalFFT,axes=(0,))
svaSignalPsd = np.abs(svaSignalFFTShift)**2
svaSignalPsdNormalized = svaSignalPsd/np.amax(svaSignalPsd,axis=0)[None,:]
svaSpectralEstimator = np.amin(svaSignalPsdNormalized,axis=1)
svaSpectralEstimatordB = 10*np.log10(svaSpectralEstimator)

""" SVA Optimized"""
received_signal_sva = np.squeeze(received_signal)
osrFact = spectrumGridOSRFact
numFFTOSR = spectrumGridOSRFact*num_samples
signalFFT = np.fft.fft(received_signal_sva,n=len(digital_freq_grid),axis=0)/num_samples # Is normalization required here for sva
Xk = signalFFT
kmKInd = np.arange(0,numFFTOSR) - osrFact
kmKInd[kmKInd<0] += numFFTOSR
XkmK = Xk[kmKInd]
kpKInd = np.arange(0,numFFTOSR) + osrFact
kpKInd[kpKInd>numFFTOSR-1] -= numFFTOSR
XkpK = Xk[kpKInd]
alphaK = np.real(Xk/(XkmK+XkpK))
alphaK[alphaK<0] = 0
alphaK[alphaK>0.5] = 0.5
svaspectrum = Xk - alphaK*(XkmK+XkpK)
svaOptimalSpectrum = np.fft.fftshift(svaspectrum)
svaOptimalSpectrum = 20*np.log10(np.abs(svaOptimalSpectrum))
svaOptimalSpectrum -= np.amax(svaOptimalSpectrum)

""" MUSIC"""
pseudo_spectrum = spec_est.music_backward(received_signal, num_sources, corr_mat_model_order, digital_freq_grid)
pseudo_spectrum = pseudo_spectrum/np.amax(pseudo_spectrum)


plt.figure(1,figsize=(20,10))
plt.title('Spatially Variant Apodization')
plt.plot(angleGrid, magnitude_spectrum_fft, label = 'FFT with rectangular window (raised cosine with alpha=0)')
plt.plot(angleGrid, magnitude_spectrum_fft_hann, label = 'FFT with hanning window (raised cosine with alpha=0.5)')
plt.plot(angleGrid, svaSpectralEstimatordB, label = 'Spatially Variant Apodization',lw=6,alpha=0.5)
plt.plot(angleGrid, svaOptimalSpectrum, label = 'Spatially Variant Apodization - optimized',color='k')
plt.vlines(-source_angle_deg,-80,20, alpha=0.3,label = 'Ground truth')
plt.xlabel('Angle(deg)')
plt.legend()
plt.grid(True)

plt.figure(2,figsize=(20,10))
plt.title('Spatially Variant Apodization vs MUSIC')
plt.plot(angleGrid, magnitude_spectrum_fft, label = 'FFT with rectangular window (raised cosine with alpha=0)')
plt.plot(angleGrid, magnitude_spectrum_fft_hann, label = 'FFT with hanning window (raised cosine with alpha=0.5)')
plt.plot(angleGrid, svaSpectralEstimatordB, label = 'Spatially Variant Apodization',lw=6,alpha=0.5)
plt.plot(angleGrid, svaOptimalSpectrum, label = 'Spatially Variant Apodization - optimized',color='k')
plt.plot(angleGrid, 10*np.log10(pseudo_spectrum), label='MUSIC')
plt.vlines(-source_angle_deg,-80,20, alpha=0.3,label = 'Ground truth')
plt.xlabel('Angle(deg)')
plt.legend()
plt.grid(True)















