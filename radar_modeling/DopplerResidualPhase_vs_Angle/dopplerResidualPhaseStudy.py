# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 22:22:48 2023

@author: Sai Gunaranjan
"""

"""
Fix bug in generation and application of Doppler correction

In this script, I have fixed a bug in the generation and application of the Doppler correction phase term for the MIMO syntesis.
This debug helped me to thoroughly understand the Doppler phase correction. The point to understand is that a true Doppler bin
doesn't have to be an integer bin. By doing an FFT and picking a peak, we can only sample integer bins. So, this fractional error
i.e true Doppler bin (which is a float) - detected Doppler bin (which is necessarily an integer) leads to an uncompensated phase error
in the MIMO array leading to angle error and SLLs. If the true Doppler bin lies exactly on a integer grid(with no fractional),
then the Doppler correction phase term becomes 1 and hence no doppler correction phase is required. Eg: If Vtrue/vRes is an integer,
then the DCM becomes unity (by chance!). But we have no control on the true velcoity of the target and hence it might or might not fall
on a bin. So we need to get as close to the true Doppler bin (float) as possible. This is achieved through an oversampled FFT evaluation.
Let us undersatnd this more closely. Consider a case where we have 64 ramps per Tx and 4 Txs. So the velres = Fc/64 * lamda/2.
The DCM becomes exp (1j*2*pi*k/64*n*64), where n is the Tx number and n varies from 0 to Ntx-1; k is Doppler bin = v/vRes.
Now, if k is exactly an integer and detection also picks up the same integer when 64 point Doppler FFT is performed on
the 64 chirp samples, then, k becomes an integer and the DCM becomes unity (64 cancels out in numerator and denominator, k and n take integer values). In this case,
luckily/coincidentally, no DCM is required while synthesizing the ULA. But consider a case when k is not an integer.
Now performing a Doppler FFT and picking the peak gives us a qunatized version of the k i.e we get int(k). As seen earlier,
if k is an integer (on the 64 scale), then the DCM terms becomes unity and this essentially means we are not applying any DCM.
But because of the error between the true Doppler bin and the quantized Doppler bin(which we call as the Doppler fractional error),
there is a Doppler residual phase on the MIMO which is un corrected. This leads to the ULA phase have phase jumps leading to
angle error and poor SLLs in the angle spectrum. To reduce this Doppler fractional error, we need to come as close to the true
Dopplr frequency as possible. This is achieved by using a oversampled FFT to obatin the Doppler bin (on the oversampled scale).
Since an oversampled FFT gives a more accurate estimate of the true Doppler frequency, we can get closer to the true Doppler frequency.
So, in the DCM phase term, the denominator is no longer 64 but may be 512 or 1024 depending on what point Doppler FFT we are performing.
So now, the numerator (64) and denominator do not cancel out anymore and also the Doppler bin 'k' is no more on the 64 scale but
it is on the oversampled FFT scale. The greater the oversampling factor (for the FFT computation), the closer we can get to the
true Doppler frequency and smaller the DCM error. But upto what point do we need to go? In other words what is the
oversampling FFT factor upto which the DCM error is tolerable. This we can get through a Monte Carlo simulation run.


"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

plt.close('all')

np.random.seed(0)

""" if flag_RxsSeparatedbyLambdaby2 = True, IC config with Rxs separated by lambda/2 and Txs separated by 2 lambda is used.
Else,  IC config with Rxs separated by 2 lambda and Txs separated by lambda/2 is used."""
flag_RxsSeparatedbyLambdaby2 = False


if flag_RxsSeparatedbyLambdaby2:
    print("\n\nIC config with Rxs separated by lambda/2 and Txs separated by 2 lambda is used\n\n")
else:
    print("\n\nIC config with Rxs separated by 2 lambda and Txs separated by lambda/2 is used\n\n")


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

numADCSamp = 1024#2048
numRx = 4
numTx = 4
numRangeSamp = numADCSamp//2
numChirpsPerTx = numChirpsDetSegment//numTx

numMIMOChannels = numTx * numRx
angleWindow = np.hanning(numMIMOChannels)#np.kaiser(numMIMOChannels, beta=8)

rangeRes = lightSpeed/(2*chirpBW)
velRes = (chirpSamplingRate/numChirpsPerTx) * (wavelength/2)

numAngleBins = 1024
angleAxis = np.arcsin((np.arange(-numAngleBins//2, numAngleBins//2))*(spatialFs/numAngleBins))*180/np.pi
angularRes = np.arcsin(spatialFs/numMIMOChannels)*180/np.pi

maxRange = rangeRes*np.arange(numRangeSamp)
maxBaseBandVelocity = (chirpSamplingRate/2) * (wavelength/2)

noiseFloorPerBindBFs = -90
totalNoisePowerdBFs = noiseFloorPerBindBFs + 10*np.log10(numADCSamp)
totalNoisePower = 10**(totalNoisePowerdBFs/10)
noiseSigma = np.sqrt(totalNoisePower)

targetRange = 35
rangeBin = targetRange/rangeRes #512
targetVelocity = np.random.uniform(0,maxBaseBandVelocity)
targetVelocityBin = targetVelocity/velRes
doppBin = targetVelocityBin # 1

targetAnglesDeg = np.array([10])
numTargets = len(targetAnglesDeg)
targetAnglesRad = (targetAnglesDeg/180) * np.pi
phaseDelta = (2*np.pi*mimoSpacing*np.sin(targetAnglesRad))/wavelength

TargetSNR = 20#15 # dB. Set to a high value so that SNR doesnt limit the Doppler phase error
RCSdelta = 60#20 # dB
antennaPatternInducedPowerDelta = 20 # dB
strongTargetSNR = TargetSNR + RCSdelta + antennaPatternInducedPowerDelta
snrPerBin = np.array([TargetSNR])
signalPowerdBFs = noiseFloorPerBindBFs + snrPerBin
signalPower = 10**(signalPowerdBFs/10)
signalAmp = np.sqrt(signalPower)
signalPhase = np.exp(1j*np.random.uniform(-np.pi,np.pi,1))
signalPhasor = signalAmp*signalPhase

numMonteCarlo = 50#50

rangeSignal = np.exp(1j*2*np.pi*rangeBin*np.arange(numADCSamp)/numADCSamp)
dopplerSignal = np.exp(1j*2*np.pi*doppBin*np.arange(numChirpsPerTx)/numChirpsPerTx)
angleSignal = np.exp(1j*phaseDelta[:,None]*np.arange(numMIMOChannels)[None,:])

angleSignal3d = np.transpose((angleSignal.reshape(numTargets,numTx,numRx)), (0,2,1)) # 2, 4,18 # 4 Rxs each separated by lamda/2 and 4Txs each separated by 2lamda
dopplerPhaseAcrossTxs = np.exp(1j*2*np.pi*doppBin*numChirpsPerTx*np.arange(numTx)/numChirpsPerTx)

if flag_RxsSeparatedbyLambdaby2:
    signal = signalPhasor[:,None, None, None, None] * rangeSignal[None,:,None,None,None] * dopplerSignal[None,None,:,None,None] * angleSignal3d[:,None,None,:,:] * dopplerPhaseAcrossTxs[None,None,None,None,:]
else:
    signal = signalPhasor[:,None, None, None, None] * rangeSignal[None,:,None,None,None] * dopplerSignal[None,None,:,None,None] * angleSignal3d[:,None,None,:,:] * dopplerPhaseAcrossTxs[None,None,None,:,None]

signal = np.sum(signal,axis=0)

noise = (noiseSigma/np.sqrt(2))*np.random.randn(numADCSamp*numChirpsPerTx*numRx*numTx*numMonteCarlo) \
    + 1j*(noiseSigma/np.sqrt(2))*np.random.randn(numADCSamp*numChirpsPerTx*numRx*numTx*numMonteCarlo)

noise = noise.reshape(numADCSamp,numChirpsPerTx,numRx,numTx,numMonteCarlo)

receivedSignal = signal[:,:,:,:,None] + noise
receivedSignal = receivedSignal*np.blackman(numADCSamp)[:,None,None,None,None]
rfft = (np.fft.fft(receivedSignal,axis=0)/numADCSamp)[0:numRangeSamp,:,:,:,:]

detectedRangeBin = np.round(rangeBin).astype('int32')
dopplerSamples = rfft[detectedRangeBin,:,:,:,:]

dopplerOsrArray = 2**np.arange(0,7,1) ## Change this OSR factor from 1 to say 16 to see the impact of the DCM on the ULA phase
numOsr = len(dopplerOsrArray)
count = 0
for dopplerOSR in dopplerOsrArray:
    numDoppFFT = dopplerOSR*numChirpsPerTx #1024
    dfft = np.fft.fft(dopplerSamples,axis=0,n=numDoppFFT)/numChirpsPerTx
    dfftEnergy = np.mean(np.abs(dfft)**2,axis=(1,2))
    detectedDopplerBinDetSeg = np.argmax(dfftEnergy,axis=0)
    detectedDopplerBin = detectedDopplerBinDetSeg
    mimoCoeff = dfft[detectedDopplerBin,:,:,np.arange(numMonteCarlo)]
    mimoCoeff = np.transpose(mimoCoeff,(1,2,0)) # Rx, Tx, numMontecarlo

    if flag_RxsSeparatedbyLambdaby2:
        doppCorrMimoCoeff = mimoCoeff*np.conj(dopplerPhaseAcrossTxs)[None,:,None]
    else:
        doppCorrMimoCoeff = mimoCoeff*np.conj(dopplerPhaseAcrossTxs)[:,None,None]

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

    if flag_RxsSeparatedbyLambdaby2:
        doppCorrMimoCoeff_inaccurateDoppler = mimoCoeff*np.conj(dopplerCorrection)[None,:,:]
    else:
        doppCorrMimoCoeff_inaccurateDoppler = mimoCoeff*np.conj(dopplerCorrection)[:,None,:]

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

    angleSpectrumMean_inaccurateDoppler = np.mean(np.abs(angleFFT_inaccurateDoppler)**2,axis=0)
    angleSpectrumMean_inaccurateDoppler = 10*np.log10(np.abs(angleSpectrumMean_inaccurateDoppler))
    angleSpectrumMean_inaccurateDoppler = angleSpectrumMean_inaccurateDoppler - np.amax(angleSpectrumMean_inaccurateDoppler)
    angleSpectrumMean_inaccurateDoppler = np.fft.fftshift(angleSpectrumMean_inaccurateDoppler)

    rfftFlatten = rfft.reshape(numRangeSamp,numChirpsPerTx*numTx*numRx*numMonteCarlo)
    rfftPowSpec = np.mean(np.abs(rfftFlatten)**2,axis=1)
    rfftPowSpecdBm = 10*np.log10(rfftPowSpec) + 10

    # plt.figure(1,figsize=(20,10))
    # plt.title('Range spectrum (dBm)')
    # plt.plot(rfftPowSpecdBm)
    # plt.xlabel('range Bins')
    # plt.ylabel('dBm')
    # plt.grid('True')

    plt.figure(count+1,figsize=(20,10))
    plt.suptitle('NumChirpsPerTx = {}, Doppler OSR = {}, numDopp FFT = {}'.format(numChirpsPerTx,dopplerOSR,numDoppFFT))
    plt.subplot(1,2,1)
    plt.title('Angle Phase')
    plt.plot(anglePhaseDeg,'-o',label='ground truth ULA phase')
    plt.plot(anglePhaseDegInaccDoppler,'-o',label='DCM corrected ULA phase')
    plt.ylabel('angle(deg)')
    plt.xlabel('MIMO channel number')
    plt.grid('True')
    plt.legend()

    plt.subplot(1,2,2)
    plt.title('Angle Spectrum')
    plt.plot(angleAxis, angleSpectrum, label='ground truth spectrum')
    plt.plot(angleAxis, angleSpectrumMean_inaccurateDoppler, label='DCM corrected spectrum')
    plt.axvline(targetAnglesDeg[0],color='k')
    plt.xlabel('angle(deg)')
    plt.grid('True')
    plt.legend()

    count += 1