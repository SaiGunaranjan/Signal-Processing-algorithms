# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 18:30:18 2023

@author: Sai Gunaranjan
"""

"""
Analyse the SLLs of Spatially variant Apodization. Looks like the SLLs of a SVA increase with signal length.
So, larger the signal length, better the SLLS. If we observe more closely, for every doubling of the signal length,
the SLLS improve by 10*np.log10(2**2) = 6 dB ! This makes sense, because, the SVA operates on the signal samples
and we know that for every doubling of the signal length, the signal power improves by 6 dB.
"""


import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..//')
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

osrFact = 32#16 #128
numSamplesArray = np.array([32,64,128,256,512,1024,2048,4096]) # 2**(np.arange(5,13,1))
signalLengths = len(numSamplesArray)


snrArr = np.array([90])#np.arange(10,90,20)
numSNR = len(snrArr)
numMC = 100
percentile = 98
beamWidthPointdB = 3

angleSLLMatrixRect_percentile = np.zeros((numSNR,signalLengths))
angleSLLMatrixHann_percentile = np.zeros((numSNR,signalLengths))
angleSLLMatrixSVA_percentile = np.zeros((numSNR,signalLengths))

beamWidthMatrixRect_percentile = np.zeros((numSNR,signalLengths))
beamWidthMatrixHann_percentile = np.zeros((numSNR,signalLengths))
beamWidthMatrixSVA_percentile = np.zeros((numSNR,signalLengths))

apodSpectrumArr = []
hannSpectrumArr = []
rectSpectrumArr = []

count_numSamp = 0
for num_samples in numSamplesArray:
    numFFTOSR = osrFact*num_samples
    totalNoisePower = noisePower_perBin*num_samples # sigmasquare totalNoisePower
    noise_sigma = np.sqrt(totalNoisePower)
    count_snrMC = 0
    for object_snr in snrArr:
        signalPowerdBFs = noiseFloor_perBin + object_snr
        signalPower = 10**(signalPowerdBFs/10)
        signalAmplitude = np.sqrt(signalPower)
        signalPhase = np.exp(1j*np.random.uniform(-np.pi, np.pi,size = num_sources))
        complex_signal_amplitudes = signalAmplitude*signalPhase

        random_freq = np.random.uniform(low=-np.pi, high=np.pi, size = 1)
        source_freq = np.pi/4#np.array([random_freq])
        source_angle_deg = np.arcsin((source_freq/(2*np.pi))*fsSpatial)*180/np.pi

        source_signals = np.matmul(np.exp(-1j*np.outer(np.arange(num_samples),source_freq)),complex_signal_amplitudes[:,None])
        angleSllArrayRect = np.empty([0])
        angleSllArrayHann = np.empty([0])
        angleSllArraySVA = np.empty([0])

        beamWidthRect = np.empty([0])
        beamWidthHann = np.empty([0])
        beamWidthSVA = np.empty([0])

        for mcarlo in range(numMC):
            wgn_noise = np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape)
            received_signal = source_signals + 0*wgn_noise

            magnitude_spectrum_fft = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(received_signal,axis=0, n=numFFTOSR)/received_signal.shape[0],axes=0)))
            magnitude_spectrum_fft -= np.amax(magnitude_spectrum_fft,axis=0)[None,:]
            magnitude_spectrum_fft = magnitude_spectrum_fft[:,0]

            """ Beamwidth computation Rect"""
            magnitude_spectrum_fft_rectOffsetAbs = np.abs(magnitude_spectrum_fft + beamWidthPointdB)
            localminIndRect = argrelextrema(magnitude_spectrum_fft_rectOffsetAbs,np.less_equal,order=1)[0]
            sortedLocalMinValRect = np.sort(magnitude_spectrum_fft_rectOffsetAbs[localminIndRect])
            binDeltaRect = np.abs(np.where(magnitude_spectrum_fft_rectOffsetAbs == sortedLocalMinValRect[0])[0][0] - np.where(magnitude_spectrum_fft_rectOffsetAbs == sortedLocalMinValRect[1])[0][0])
            binDeltaRectNativeScale = (binDeltaRect/numFFTOSR) * num_samples
            beamWidthRect = np.hstack((beamWidthRect,binDeltaRectNativeScale))

            magnitude_spectrum_fft_hann = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(received_signal*np.hanning(num_samples)[:,None],axis=0, n=numFFTOSR)/received_signal.shape[0],axes=0)))
            magnitude_spectrum_fft_hann -= np.amax(magnitude_spectrum_fft_hann,axis=0)[None,:]
            magnitude_spectrum_fft_hann = magnitude_spectrum_fft_hann[:,0]

            """ Beamwidth computation Hanning"""
            magnitude_spectrum_fft_hannOffsetAbs = np.abs(magnitude_spectrum_fft_hann + beamWidthPointdB)
            localminIndHann = argrelextrema(magnitude_spectrum_fft_hannOffsetAbs,np.less_equal,order=1)[0]
            sortedLocalMinValHann = np.sort(magnitude_spectrum_fft_hannOffsetAbs[localminIndHann])
            binDeltaHann = np.abs(np.where(magnitude_spectrum_fft_hannOffsetAbs == sortedLocalMinValHann[0])[0][0] - np.where(magnitude_spectrum_fft_hannOffsetAbs == sortedLocalMinValHann[1])[0][0])
            binDeltaHannNativeScale = (binDeltaHann/numFFTOSR) * num_samples
            beamWidthHann = np.hstack((beamWidthHann,binDeltaHannNativeScale))

            """ SVA Optimized"""
            svaOptimalComplexSpectrumfftshifted, svaOptimalMagSpectrumdB = spatially_variant_apodization_optimized(received_signal, osrFact)
            """ Beamwidth computation SVA"""
            svaOptimalMagSpectrumdBOffsetAbs = np.abs(svaOptimalMagSpectrumdB + beamWidthPointdB)
            localminIndSVA = argrelextrema(svaOptimalMagSpectrumdBOffsetAbs,np.less_equal,order=1)[0]
            sortedLocalMinValSVA = np.sort(svaOptimalMagSpectrumdBOffsetAbs[localminIndSVA])
            binDeltaSVA = np.abs(np.where(svaOptimalMagSpectrumdBOffsetAbs == sortedLocalMinValSVA[0])[0][0] - np.where(svaOptimalMagSpectrumdBOffsetAbs == sortedLocalMinValSVA[1])[0][0])
            binDeltaSVANativeScale = (binDeltaSVA/numFFTOSR) * num_samples
            beamWidthSVA = np.hstack((beamWidthSVA,binDeltaSVANativeScale))



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

        rectSpectrumArr.append(magnitude_spectrum_fft)
        hannSpectrumArr.append(magnitude_spectrum_fft_hann)
        apodSpectrumArr.append(svaOptimalMagSpectrumdB)


        angleSLLMatrixRect_percentile[count_snrMC,count_numSamp] = np.percentile(angleSllArrayRect,percentile)
        angleSLLMatrixHann_percentile[count_snrMC,count_numSamp] = np.percentile(angleSllArrayHann,percentile)
        angleSLLMatrixSVA_percentile[count_snrMC,count_numSamp] = np.percentile(angleSllArraySVA,percentile)

        beamWidthMatrixRect_percentile[count_snrMC,count_numSamp] = np.percentile(beamWidthRect,percentile)
        beamWidthMatrixHann_percentile[count_snrMC,count_numSamp] = np.percentile(beamWidthHann,percentile)
        beamWidthMatrixSVA_percentile[count_snrMC,count_numSamp] = np.percentile(beamWidthSVA,percentile)

        count_snrMC += 1

    count_numSamp += 1

n = 0

# numSamplesArrayLeg = ['Sig Len = ' + str(ele) for ele in numSamplesArray]
# plt.figure(n+1,figsize=(20,10), dpi=200)
# plt.title('{} ile Angle SLLs(dBc) vs SNR'.format(percentile))
# plt.plot(snrArr, angleSLLMatrixSVA_percentile, '-o')
# plt.xlabel('SNR (dB)')
# plt.ylabel('SLL (dBc)')
# plt.grid(True)
# plt.legend(numSamplesArrayLeg)
# plt.ylim([-80,10])
# n+=1


# snrArrLeg = ['SNR = ' + str(ele) + 'dB' for ele in snrArr]
# plt.figure(n+1,figsize=(20,10), dpi=200)
# plt.title('{} ile Angle SLLs(dBc) vs Signal Length'.format(percentile))
# plt.plot(numSamplesArray, angleSLLMatrixSVA_percentile.T, '-o')
# plt.xlabel('Signal Length')
# plt.ylabel('SLL (dBc)')
# plt.grid(True)
# plt.legend(snrArrLeg)
# plt.xticks(numSamplesArray)
# # plt.xscale("log")
# plt.ylim([-80,10])
# n+=1


plt.figure(n+1,figsize=(20,10),dpi=200)
plt.title('{} %ile SLLs(dBc) vs Signal Length'.format(percentile))
plt.plot(numSamplesArray, angleSLLMatrixRect_percentile[-1,:], '-o',label='Rect Window')
plt.plot(numSamplesArray, angleSLLMatrixHann_percentile[-1,:], '-o',label='Hanning Window')
plt.plot(numSamplesArray, angleSLLMatrixSVA_percentile[-1,:], '-o', label='SVA')
plt.xlabel('Signal Length')
plt.ylabel('SLL (dBc)')
plt.grid(True)
plt.legend()
plt.xticks(numSamplesArray)
plt.xscale("log")
plt.ylim([-80,0])
n+=1

plt.figure(n+1,figsize=(20,10),dpi=200)
plt.title('{} %ile {} db main lobe (bins) vs Signal Length'.format(percentile, beamWidthPointdB))
plt.plot(numSamplesArray, beamWidthMatrixRect_percentile[-1,:], '-o',lw=4,alpha=0.5,label='Rect Window')
plt.plot(numSamplesArray, beamWidthMatrixHann_percentile[-1,:], '-o',label='Hanning Window')
plt.plot(numSamplesArray, beamWidthMatrixSVA_percentile[-1,:], '-+', color = 'red',label='SVA')
plt.xlabel('Signal Length')
plt.ylabel('bins')
plt.grid(True)
plt.legend()
plt.xticks(numSamplesArray)
plt.xscale("log")
plt.ylim([0,3])
n+=1


plt.figure(n+1,figsize=(20,10))
plt.suptitle('Spectrum(oversampled) for different signal lengths')
for ele in range(signalLengths):
    plt.subplot(2,4,ele+1)
    plt.title('Signal length = {}'.format(numSamplesArray[ele]))
    plt.plot(rectSpectrumArr[ele], label='Rectangular')
    plt.plot(hannSpectrumArr[ele], label='Hanning')
    plt.plot(apodSpectrumArr[ele], color='k', label='Apodization')
    plt.ylim(-90,5)
    maxInd = np.argmax(apodSpectrumArr[ele])
    plt.xlim([maxInd-100, maxInd+100])
    plt.grid(True)
    plt.legend()











