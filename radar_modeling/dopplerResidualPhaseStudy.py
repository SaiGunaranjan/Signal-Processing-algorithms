# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 22:22:48 2023

@author: Sai Gunaranjan
"""


import numpy as np
import matplotlib.pyplot as plt
# from scipy import signal

plt.close('all')

np.random.seed(0)

"""Chirp Parameters """
chirpCenterFreq = 76.5e9 # Hz
chirpBW = 500e6 # Hz
lightSpeed = 3e8
interChirpTime = 44e-6 # sec
chirpSamplingRate = 1/interChirpTime
totalChirps = 256
numChirpsDetSegment = totalChirps//2
wavelength = lightSpeed/chirpCenterFreq
mimoSpacing = 2e-3 #m
spatialFs = wavelength/mimoSpacing

rangeRes = lightSpeed/(2*chirpBW)
detSegmentVelRes = (chirpSamplingRate/numChirpsDetSegment) * (wavelength/2)

numADCSamp = 512#2048
numRx = 4
numTx = 4
numRangeSamp = numADCSamp//2
numChirpsPerTx = numChirpsDetSegment//numTx

numMIMOChannels = numTx * numRx

numAngleBins = 256#numMIMOChannels
angleAxis = np.arcsin((np.arange(-numAngleBins//2, numAngleBins//2))*(spatialFs/numAngleBins))*180/np.pi
angularRes = np.arcsin(spatialFs/numMIMOChannels)*180/np.pi


maxRange = rangeRes*np.arange(numRangeSamp)
maxBaseBandVelocity = (chirpSamplingRate/2) * (wavelength/2)

noiseFloorPerBindBFs = -90
totalNoisePowerdBFs = noiseFloorPerBindBFs + 10*np.log10(numADCSamp)
totalNoisePower = 10**(totalNoisePowerdBFs/10)
noiseSigma = np.sqrt(totalNoisePower)

targetRange = 35#np.random.uniform(10,maxRange-10)
rangeBin = targetRange/rangeRes #512
targetVelocity = np.random.uniform(0,maxBaseBandVelocity)
targetVelocityBinDetSegScale = targetVelocity/detSegmentVelRes
doppBin = targetVelocityBinDetSegScale # 1


targetAnglesDeg = np.array([45])
numTargets = len(targetAnglesDeg)
targetAnglesRad = (targetAnglesDeg/180) * np.pi
phaseDelta = (2*np.pi*mimoSpacing*np.sin(targetAnglesRad))/wavelength


TargetSNR = 20#15 # dB. Change to -9 dB so that we get  18 dB SNR at detection
RCSdelta = 60#20 # dB
antennaPatternInducedPowerDelta = 20 # dB
strongTargetSNR = TargetSNR + RCSdelta + antennaPatternInducedPowerDelta
# snrPerBin = np.array([weakTargetSNR, strongTargetSNR])
snrPerBin = np.array([TargetSNR])
signalPowerdBFs = noiseFloorPerBindBFs + snrPerBin
signalPower = 10**(signalPowerdBFs/10)
signalAmp = np.sqrt(signalPower)
signalPhase = np.exp(1j*np.random.uniform(-np.pi,np.pi,2))
signalPhasor = signalAmp*signalPhase

numMonteCarlo = 50#50

rangeSignal = np.exp(1j*2*np.pi*rangeBin*np.arange(numADCSamp)/numADCSamp)
dopplerSignal = np.exp(1j*2*np.pi*doppBin*np.arange(numChirpsPerTx)/numChirpsDetSegment)
angleSignal = np.exp(1j*phaseDelta[:,None]*np.arange(numMIMOChannels)[None,:])


angleSignal3d = np.transpose((angleSignal.reshape(numTargets,numTx,numRx)), (0,2,1)) # 2, 4,18 # 4 Rxs each separated by lamda/2 and 4Txs each separated by 2lamda
dopplerPhaseAcrossTxs = np.exp(1j*2*np.pi*doppBin*numChirpsPerTx*np.arange(numTx)/numChirpsDetSegment)

signal = signalPhasor[:,None, None, None, None] * rangeSignal[None,:,None,None,None] * dopplerSignal[None,None,:,None,None] * angleSignal3d[:,None,None,:,:] * dopplerPhaseAcrossTxs[None,None,None,None,:]
signal = np.sum(signal,axis=0)

noise = (noiseSigma/np.sqrt(2))*np.random.randn(numADCSamp*numChirpsPerTx*numRx*numTx*numMonteCarlo) \
    + 1j*(noiseSigma/np.sqrt(2))*np.random.randn(numADCSamp*numChirpsPerTx*numRx*numTx*numMonteCarlo)

noise = noise.reshape(numADCSamp,numChirpsPerTx,numRx,numTx,numMonteCarlo)


receivedSignal = signal[:,:,:,:,None] + noise
receivedSignal = receivedSignal*np.blackman(numADCSamp)[:,None,None,None,None]
rfft = (np.fft.fft(receivedSignal,axis=0)/numADCSamp)[0:numRangeSamp,:,:,:,:]


detectedRangeBin = np.round(rangeBin).astype('int32')

dopplerSamples = rfft[detectedRangeBin,:,:,:,:]
dfft = np.fft.fft(dopplerSamples,axis=0,n=numChirpsDetSegment)/numChirpsPerTx
detectedDopplerBinDetSeg = (np.round(targetVelocityBinDetSegScale).astype('int32'))
detectedDopplerBin = detectedDopplerBinDetSeg
mimoCoeff = dfft[detectedDopplerBin,:,:,:]

angleWindow = np.hanning(numMIMOChannels)#np.kaiser(numMIMOChannels, beta=8)

doppCorrMimoCoeff = mimoCoeff*np.conj(dopplerPhaseAcrossTxs)[None,:,None]
doppCorrMimoCoeffFlatten = np.transpose(doppCorrMimoCoeff,(2,1,0)).reshape(numMonteCarlo,numMIMOChannels)
doppCorrMimoCoeffFlatten = doppCorrMimoCoeffFlatten*angleWindow[None,:]
angleFFT = np.fft.fft(doppCorrMimoCoeffFlatten,axis=1,n=numAngleBins)/numMIMOChannels
angleSpectrumMean = np.mean(np.abs(angleFFT)**2,axis=0)
angleSpectrum = 10*np.log10(np.abs(angleSpectrumMean))
angleSpectrum = angleSpectrum - np.amax(angleSpectrum)
angleSpectrum = np.fft.fftshift(angleSpectrum)

dopplerCorrection = np.exp(1j*2*np.pi*detectedDopplerBin*numChirpsPerTx*np.arange(numTx)/numChirpsDetSegment)
# dopplerCorrection = np.exp(1j*2*np.pi*doppBin*numChirpsPerTx*np.arange(numTx)/numChirpsDetSegment)

doppCorrMimoCoeff_inaccurateDoppler = mimoCoeff*np.conj(dopplerCorrection)[None,:,None]
doppCorrMimoCoeffFlatten_inaccurateDoppler = np.transpose(doppCorrMimoCoeff_inaccurateDoppler,(2,1,0)).reshape(numMonteCarlo,numMIMOChannels)
doppCorrMimoCoeffFlatten_inaccurateDoppler = doppCorrMimoCoeffFlatten_inaccurateDoppler*angleWindow[None,:]
angleFFT_inaccurateDoppler = np.fft.fft(doppCorrMimoCoeffFlatten_inaccurateDoppler,axis=1,n=numAngleBins)/numMIMOChannels
angleSpectrumMean_inaccurateDoppler = np.mean(np.abs(angleFFT_inaccurateDoppler)**2,axis=0)
angleSpectrum_inaccurateDoppler = 10*np.log10(np.abs(angleSpectrumMean_inaccurateDoppler))
angleSpectrum_inaccurateDoppler = angleSpectrum_inaccurateDoppler - np.amax(angleSpectrum_inaccurateDoppler)
angleSpectrum_inaccurateDoppler = np.fft.fftshift(angleSpectrum_inaccurateDoppler)


rfftFlatten = rfft.reshape(numRangeSamp,numChirpsPerTx*numTx*numRx*numMonteCarlo)
rfftPowSpec = np.mean(np.abs(rfftFlatten)**2,axis=1)
rfftPowSpecdBm = 10*np.log10(rfftPowSpec) + 10


anglePhaseDeg = np.unwrap(np.angle(doppCorrMimoCoeffFlatten),axis=1)*180/np.pi
anglePhaseDeg = np.mean(anglePhaseDeg,axis=0)
anglePhaseDegInaccDoppler = np.unwrap(np.angle(doppCorrMimoCoeffFlatten_inaccurateDoppler),axis=1)*180/np.pi
anglePhaseDegInaccDoppler = np.mean(anglePhaseDegInaccDoppler,axis=0)

plt.figure(1,figsize=(20,10))
plt.title('Range spectrum (dBm)')
plt.plot(rfftPowSpecdBm,label='without compression/decompression')
plt.xlabel('range Bins')
plt.ylabel('dBm')
plt.grid('True')
plt.legend()


plt.figure(2,figsize=(20,10))
plt.title('Angle Phase')
plt.plot(anglePhaseDeg,'-o',label='ground truth Phase')
plt.plot(anglePhaseDegInaccDoppler,'-o',label='with inaccurate DCM')
plt.ylabel('angle(deg)')
plt.xlabel('MIMO channel number')
plt.grid('True')
plt.legend()


plt.figure(3,figsize=(20,10))
plt.title('Angle Spectrum')
plt.plot(angleAxis, angleSpectrum, label='ground truth spectrum')
plt.plot(angleAxis, angleSpectrum_inaccurateDoppler, label='with inaccurate DCM')
for ele in np.arange(numTargets):
    plt.axvline(targetAnglesDeg[ele],color='k')

plt.xlabel('angle(deg)')
plt.grid('True')
plt.legend()