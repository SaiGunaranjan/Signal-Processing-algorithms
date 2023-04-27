# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 22:22:48 2023

@author: Sai Gunaranjan
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

plt.close('all')

np.random.seed(0)

"""Chirp Parameters """
chirpCenterFreq = 76.5e9 # Hz
chirpBW = 500e6 # Hz
lightSpeed = 3e8
interChirpTime = 44e-6 # sec
chirpSamplingRate = 1/interChirpTime
totalChirps = 512
numChirpsDetSegment = totalChirps//2
wavelength = lightSpeed/chirpCenterFreq
mimoSpacing = 2e-3 #m
spatialFs = wavelength/mimoSpacing



numADCSamp = 512#2048
numRx = 4
numTx = 4
numRangeSamp = numADCSamp//2
numChirpsPerTx = numChirpsDetSegment//numTx
dopplerOSR = 16
numDoppFFT = dopplerOSR*numChirpsPerTx #1024

numMIMOChannels = numTx * numRx

rangeRes = lightSpeed/(2*chirpBW)
velRes = (chirpSamplingRate/numChirpsPerTx) * (wavelength/2)

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
targetVelocityBin = targetVelocity/velRes
doppBin = targetVelocityBin # 1


targetAnglesDeg = np.array([10])
numTargets = len(targetAnglesDeg)
targetAnglesRad = (targetAnglesDeg/180) * np.pi
phaseDelta = (2*np.pi*mimoSpacing*np.sin(targetAnglesRad))/wavelength


TargetSNR = 10#15 # dB. Change to -9 dB so that we get  18 dB SNR at detection
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
dopplerSignal = np.exp(1j*2*np.pi*doppBin*np.arange(numChirpsPerTx)/numChirpsPerTx)
angleSignal = np.exp(1j*phaseDelta[:,None]*np.arange(numMIMOChannels)[None,:])


angleSignal3d = np.transpose((angleSignal.reshape(numTargets,numTx,numRx)), (0,2,1)) # 2, 4,18 # 4 Rxs each separated by lamda/2 and 4Txs each separated by 2lamda
dopplerPhaseAcrossTxs = np.exp(1j*2*np.pi*doppBin*numChirpsPerTx*np.arange(numTx)/numChirpsPerTx)

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
dfft = np.fft.fft(dopplerSamples,axis=0,n=numDoppFFT)/numChirpsPerTx
dfftEnergy = np.mean(np.abs(dfft)**2,axis=(1,2))

detectedDopplerBinDetSeg = np.argmax(dfftEnergy,axis=0)
detectedDopplerBin = detectedDopplerBinDetSeg
mimoCoeff = dfft[detectedDopplerBin,:,:,np.arange(numMonteCarlo)]
mimoCoeff = np.transpose(mimoCoeff,(1,2,0)) # Rx, Tx, numMontecarlo

angleWindow = np.hanning(numMIMOChannels)#np.kaiser(numMIMOChannels, beta=8)

doppCorrMimoCoeff = mimoCoeff*np.conj(dopplerPhaseAcrossTxs)[None,:,None]
doppCorrMimoCoeffFlatten = np.transpose(doppCorrMimoCoeff,(2,1,0)).reshape(numMonteCarlo,numMIMOChannels)
anglePhaseDeg = np.unwrap(np.angle(doppCorrMimoCoeffFlatten),axis=1)*180/np.pi
anglePhaseDeg = np.mean(anglePhaseDeg,axis=0)
doppCorrMimoCoeffFlatten = doppCorrMimoCoeffFlatten*angleWindow[None,:]
angleFFT = np.fft.fft(doppCorrMimoCoeffFlatten,axis=1,n=numAngleBins)/numMIMOChannels
angleSpectrumMean = np.mean(np.abs(angleFFT)**2,axis=0)
angleSpectrum = 10*np.log10(np.abs(angleSpectrumMean))
angleSpectrum = angleSpectrum - np.amax(angleSpectrum)
angleSpectrum = np.fft.fftshift(angleSpectrum)

dopplerCorrection = np.exp(1j*2*np.pi*detectedDopplerBin[None,:]*numChirpsPerTx*np.arange(numTx)[:,None]/numDoppFFT)
# dopplerCorrection = np.exp(1j*2*np.pi*doppBin*numChirpsPerTx*np.arange(numTx)/numDoppFFT)

doppCorrMimoCoeff_inaccurateDoppler = mimoCoeff*np.conj(dopplerCorrection)[None,:,:]
doppCorrMimoCoeffFlatten_inaccurateDoppler = np.transpose(doppCorrMimoCoeff_inaccurateDoppler,(2,1,0)).reshape(numMonteCarlo,numMIMOChannels)
anglePhaseDegInaccDoppler = np.unwrap(np.angle(doppCorrMimoCoeffFlatten_inaccurateDoppler),axis=1)*180/np.pi
anglePhaseDegInaccDoppler = np.mean(anglePhaseDegInaccDoppler,axis=0)
doppCorrMimoCoeffFlatten_inaccurateDoppler = doppCorrMimoCoeffFlatten_inaccurateDoppler*angleWindow[None,:]
angleFFT_inaccurateDoppler = np.fft.fft(doppCorrMimoCoeffFlatten_inaccurateDoppler,axis=1,n=numAngleBins)/numMIMOChannels
angleFFT_inaccurateDopplerfftShifted = np.fft.fftshift(angleFFT_inaccurateDoppler,axes=(1,))
angleSpectrum_inaccurateDopplerfftShifted = np.abs(angleFFT_inaccurateDopplerfftShifted)**2
angleSpectrum_inaccurateDopplerfftShifted_dB = 10*np.log10(angleSpectrum_inaccurateDopplerfftShifted)
angleSpectrum_inaccurateDopplerfftShifted_dB -= np.amax(angleSpectrum_inaccurateDopplerfftShifted_dB,axis=1)[:,None]

""" Angle Error std"""
objAngIndex = np.argmax(angleSpectrum_inaccurateDopplerfftShifted,axis=1)
objAngleDeg = angleAxis[objAngIndex]
angleErrorDeg = targetAnglesDeg - objAngleDeg
angleErrorStd = np.std(angleErrorDeg)
angleErrorStddB = 20*np.log10(angleErrorStd)

""" Angle SLLs computation"""

sllValdBc = np.zeros((numMonteCarlo),dtype=np.float32)
for ele1 in np.arange(numMonteCarlo):
    localMaxInd = argrelextrema(angleSpectrum_inaccurateDopplerfftShifted_dB[ele1,:],np.greater,axis=0,order=2)[0]
    try:
        sllInd = np.argsort(angleSpectrum_inaccurateDopplerfftShifted_dB[ele1,localMaxInd])[-2] # 1st SLL
        sllValdBc[ele1] = angleSpectrum_inaccurateDopplerfftShifted_dB[ele1,localMaxInd[sllInd]]
    except IndexError:
        sllValdBc[ele1] = 0


# angleSllArray = np.hstack((angleSllArray,sllValdBc))


angleSpectrumMean_inaccurateDoppler = np.mean(np.abs(angleFFT_inaccurateDoppler)**2,axis=0)
angleSpectrumMean_inaccurateDoppler = 10*np.log10(np.abs(angleSpectrumMean_inaccurateDoppler))
angleSpectrumMean_inaccurateDoppler = angleSpectrumMean_inaccurateDoppler - np.amax(angleSpectrumMean_inaccurateDoppler)
angleSpectrumMean_inaccurateDoppler = np.fft.fftshift(angleSpectrumMean_inaccurateDoppler)


rfftFlatten = rfft.reshape(numRangeSamp,numChirpsPerTx*numTx*numRx*numMonteCarlo)
rfftPowSpec = np.mean(np.abs(rfftFlatten)**2,axis=1)
rfftPowSpecdBm = 10*np.log10(rfftPowSpec) + 10





plt.figure(1,figsize=(20,10))
plt.title('Range spectrum (dBm)')
plt.plot(rfftPowSpecdBm)
plt.xlabel('range Bins')
plt.ylabel('dBm')
plt.grid('True')



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
plt.plot(angleAxis, angleSpectrumMean_inaccurateDoppler, label='with inaccurate DCM')
for ele in np.arange(numTargets):
    plt.axvline(targetAnglesDeg[ele],color='k')

plt.xlabel('angle(deg)')
plt.grid('True')
plt.legend()