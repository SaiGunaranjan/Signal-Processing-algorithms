# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 21:20:41 2023

@author: saiguna
"""

"""
Model impact of inaccurate range update in DDMA scheme

In this script, I have modelled and shown the impact of an inaccurate range update(per ramp)
on the angle spectrum. I have used the angle error(from ground truth) and the SLL
as a metric to asses the impact. Towards this modelling, I have swept the range bin offset
from -2 to +2 in steps of 0.125 bins around the true range bin movement across chirps.
I then plot the angle accuracy and angle spectrum SLLs as a function of the range bin offset.
As expected, the DDMA scheme is very sensitive to the range update since
we potentially update the range bin on a chirp by chirp basis and also
phase correct per chirp(if there is a bin update).
"""


""" The derivation for the DDMA scheme is available in the below location:
    https://saigunaranjan.atlassian.net/wiki/spaces/RM/pages/1966081/Code+Division+Multiple+Access+in+FMCW+RADAR"""

""" The derivation for the Doppler phase modulation due to range movement is available at the below link:
    https://saigunaranjan.atlassian.net/wiki/spaces/RM/pages/2719745/Doppler+Phase+modulation+due+to+Range+movement+across+chirps+-+Analog+signal

    https://saigunaranjan.atlassian.net/wiki/spaces/RM/pages/2949121/Doppler+Phase+modulation+due+to+residual+Range+bin+movement+across+chirps+-+Sampled+signal
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from mimoPhasorSynthesis import mimoPhasorSynth


plt.close('all')

platform = 'SRIR16' # 'SRIR16', 'SRIR256', 'SRIR144'

if (platform == 'SRIR16'):
    numTx_simult = 4 # separated by lambda/2
    numRx = 4 # separated by 2lambda
    numMIMO = 16 # All MIMO in azimuth only
    numRamps = 128 # Assuming 128 ramps for both detection and MIMO segments
elif (platform == 'SRIR144'):
    numTx_simult = 12
    numRx = 12
    numMIMO = 48
    numRamps = 140 # Assuming 140 ramps for both detection and MIMO segments
elif (platform == 'SRIR256'):
    numTx_simult = 13
    numRx = 16
    numMIMO = 74
    numRamps = 140 # Assuming 140 ramps for both detection and MIMO segments

numSamp = 2048 # Number of ADC time domain samples
numSampPostRfft = numSamp//2
numAngleFFT = 2048
mimoArraySpacing = 2e-3 # 2mm
lightSpeed = 3e8
numBitsPhaseShifter = 7
numPhaseCodes = 2**numBitsPhaseShifter
DNL = 360/(numPhaseCodes) # DNL in degrees

DoppAmbigNumArr = np.arange(-2,3) # Doppler Ambiguity number/Doppler Integer hypothesis
""" -1/+1 hypothesis is 3 times as likely as -2/2 hypothesis. 0 hypthesis is 2 times as likely as -1/+1 hypothesis """
DoppAmbNum = np.random.choice(DoppAmbigNumArr,p=[1/20, 3/20, 12/20, 3/20, 1/20])


""" Chirp Parameters"""

numDoppFFT = 2048
chirpBW = 1e9 # Hz
centerFreq = 76.5e9 # GHz
interRampTime = 44e-6 # us
rampSamplingRate = 1/interRampTime
rangeRes = lightSpeed/(2*chirpBW)
maxRange = numSampPostRfft*rangeRes # m
lamda = lightSpeed/centerFreq
""" With 30 deg, we see periodicity since 30 divides 360 but with say 29 deg, it doesn't divide 360 and hence periodicity is significantly reduced"""
phaseStepPerTx_deg = 29#29.3


Fs_spatial = lamda/mimoArraySpacing
angAxis_deg = np.arcsin(np.arange(-numAngleFFT//2, numAngleFFT//2)*(Fs_spatial/numAngleFFT))*180/np.pi


## RF parameters
thermalNoise = -174 # dBm/Hz
noiseFigure = 10 # dB
baseBandgain = 34 #dB
adcSamplingRate = 56.25e6 # 56.25 MHz
adcSamplingTime = 1/adcSamplingRate # s
chirpOnTime = numSamp*adcSamplingTime
chirpSlope = chirpBW/chirpOnTime
dBFs_to_dBm = 10
binSNR = 10#20# # dB
totalNoisePower_dBm = thermalNoise + noiseFigure + baseBandgain + 10*np.log10(adcSamplingRate)
totalNoisePower_dBFs = totalNoisePower_dBm - 10
noiseFloor_perBin = totalNoisePower_dBFs - 10*np.log10(numSamp) # dBFs/bin
noisePower_perBin = 10**(noiseFloor_perBin/10)
totalNoisePower = noisePower_perBin*numSamp # sigmasquare totalNoisePower
sigma = np.sqrt(totalNoisePower)
signalPowerdBFs = noiseFloor_perBin + binSNR
signalPower = 10**(signalPowerdBFs/10)
signalAmplitude = np.sqrt(signalPower)
signalPhase = np.exp(1j*np.random.uniform(-np.pi, np.pi))
signalphasor = signalAmplitude*signalPhase

""" Derived Parameters """
chirpSamplingRate = 1/interRampTime
maxVelBaseband_mps = (chirpSamplingRate/2) * (lamda/2) # m/s
print('Max base band velocity = {0:.2f} m/s'.format(maxVelBaseband_mps))
FsEquivalentVelocity = 2*maxVelBaseband_mps # Fs = 2*Fs/2
velocityRes = (chirpSamplingRate/numRamps) * (lamda/2)
print('Velocity resolution = {0:.2f} m/s'.format(velocityRes))


""" Target definition"""
objectRange = np.random.uniform(10,maxRange-10) # 60.3 # m
# objectVelocity_mps = np.random.uniform(-maxVelBaseband_mps-2*FsEquivalentVelocity, \
#                                         maxVelBaseband_mps+2*FsEquivalentVelocity)
objectVelocity_mps = 55#np.random.uniform(-maxVelBaseband_mps+(DoppAmbNum*FsEquivalentVelocity), \
                                        # -maxVelBaseband_mps+(DoppAmbNum*FsEquivalentVelocity)+FsEquivalentVelocity)
objectAzAngle_deg = np.random.uniform(-50,50)
objectAzAngle_rad = (objectAzAngle_deg/360) * (2*np.pi)

objectElAngle_deg = 0 # phi=0 plane angle
objectElAngle_rad = (objectElAngle_deg/360) * (2*np.pi)

mimoPhasor, mimoPhasor_txrx, ulaInd = mimoPhasorSynth(platform, lamda, objectAzAngle_rad, objectElAngle_rad)


objectVelocity_baseBand_mps = np.mod(objectVelocity_mps, FsEquivalentVelocity) # modulo Fs [from 0 to Fs]
objectVelocityBin = objectVelocity_baseBand_mps/velocityRes
objectRangeBin = objectRange/rangeRes

rangebinStepSize = 0.125
rangebinOffset = np.arange(-2,2+rangebinStepSize,0.125)
noRangeOffsetInd = np.where(rangebinOffset==0)[0][0]
numRangeOffsets = len(rangebinOffset)
rangeMoved = objectRange + objectVelocity_mps*interRampTime*np.arange(numRamps)[None,:]
rangeBinsMoved = rangeMoved/rangeRes + rangebinOffset[:,None]
rangeBinsMoved = np.floor(rangeBinsMoved).astype('int32')

phaseStepPerRamp_deg = np.arange(numTx_simult)*phaseStepPerTx_deg # Phase step per ramp per Tx
phaseStepPerRamp_rad = (phaseStepPerRamp_deg/360)*2*np.pi

phaseShifterCodes = DNL*np.arange(numPhaseCodes)
phaseShifterNoise = np.random.uniform(-DNL/2, DNL/2, numPhaseCodes)
phaseShifterCodes_withNoise = phaseShifterCodes + phaseShifterNoise

""" Ensure that the phase shifter LUT is without any bias and is from 0 to 360 degrees"""
phaseShifterCodes_withNoise = np.mod(phaseShifterCodes_withNoise,360)

rampPhaseIdeal_deg = phaseStepPerRamp_deg[:,None]*(np.arange(numRamps)[None,:])
rampPhaseIdeal_degWrapped = np.mod(rampPhaseIdeal_deg, 360)
phaseCodesIndexToBeApplied = np.argmin(np.abs(rampPhaseIdeal_degWrapped[:,:,None] - phaseShifterCodes_withNoise[None,None,:]),axis=2)
phaseCodesToBeApplied = phaseShifterCodes_withNoise[phaseCodesIndexToBeApplied]
phaseCodesToBeApplied_rad = (phaseCodesToBeApplied/180) * np.pi

rangeTerm = signalphasor*np.exp(1j*((2*np.pi*objectRangeBin)/numSamp)*np.arange(numSamp))
dopplerTerm = np.exp(1j*((2*np.pi*objectVelocityBin)/numRamps)*np.arange(numRamps))
""" Range Bin migration term"""
rangeBinMigration = \
    np.exp(1j*2*np.pi*chirpSlope*(2*objectVelocity_mps/lightSpeed)*interRampTime*adcSamplingTime*np.arange(numRamps)[:,None]*np.arange(numSamp)[None,:])

rxSignal = mimoPhasor_txrx[0,0,:]
txSignal = mimoPhasor_txrx[0,:,0]

signal_phaseCode = np.exp(1j*phaseCodesToBeApplied_rad)
phaseCodedTxSignal = dopplerTerm[None,:] * signal_phaseCode * txSignal[:,None] # [numTx, numRamps]
phaseCodedTxRxSignal = phaseCodedTxSignal[:,:,None]*rxSignal[None,None,:] #[numTx, numRamps, numTx, numRx]
phaseCodedTxRxSignal_withRangeTerm = rangeTerm[None,None,None,:] * phaseCodedTxRxSignal[:,:,:,None]
""" Including the range bin migration term as well"""
phaseCodedTxRxSignal_withRangeTerm = phaseCodedTxRxSignal_withRangeTerm * rangeBinMigration[None,:,None,:]
signal = np.sum(phaseCodedTxRxSignal_withRangeTerm, axis=(0)) # [numRamps,numRx, numSamp]
noise = (sigma/np.sqrt(2))*np.random.randn(numRamps*numRx*numSamp) + 1j*(sigma/np.sqrt(2))*np.random.randn(numRamps*numRx*numSamp)
noise = noise.reshape(numRamps, numRx, numSamp)
signal = signal + noise

signal_rangeWin = signal*np.hanning(numSamp)[None,None,:]
signal_rfft = np.fft.fft(signal_rangeWin,axis=2)/numSamp
signal_rfft = signal_rfft[:,:,0:numSampPostRfft]
signal_rfft_powermean = np.mean(np.abs(signal_rfft)**2,axis=(0,1))

rangeBinsToSample = rangeBinsMoved
chirpSamp_givenRangeBin = signal_rfft[np.arange(numRamps)[None,:],:,rangeBinsToSample]

""" Correcting for the Pi phase jump caused due to the Range bin Migration"""
binDelta = np.abs(rangeBinsToSample[:,1::] - rangeBinsToSample[:,0:-1])
tempVar = binDelta*np.pi
binMigrationPhaseCorrTerm = np.zeros((rangeBinsToSample.shape),dtype=np.float32)
binMigrationPhaseCorrTerm[:,1::] = tempVar
binMigrationPhasorCorrTerm = np.exp(-1j*np.cumsum(binMigrationPhaseCorrTerm,axis=1))
chirpSamp_givenRangeBin = chirpSamp_givenRangeBin*binMigrationPhasorCorrTerm[:,:,None]

""" Correcting for the Doppler modulation caused due to the Range bin Migration"""
rbmModulationAnalogFreq = (chirpBW/lightSpeed)*objectVelocity_mps
rbmModulationDigitalFreq = (rbmModulationAnalogFreq/rampSamplingRate)*2*np.pi
rbmModulationCorrectionTerm = np.exp(-1j*rbmModulationDigitalFreq*np.arange(numRamps))
chirpSamp_givenRangeBin = chirpSamp_givenRangeBin*rbmModulationCorrectionTerm[None,:,None]

signalWindowed = chirpSamp_givenRangeBin*np.hanning(numRamps)[None,:,None]
signalFFT = np.fft.fft(signalWindowed, axis=1, n = numDoppFFT)/numRamps
signalFFTShift = signalFFT
signalFFTShiftSpectrum = np.abs(signalFFTShift)**2
signalMagSpectrum = 10*np.log10(np.abs(signalFFTShiftSpectrum))

objectVelocityBinNewScale = (objectVelocityBin/numRamps)*numDoppFFT
binOffset_Txphase = (phaseStepPerRamp_rad/(2*np.pi))*numDoppFFT
dopplerBinsToSample = np.round(objectVelocityBinNewScale + binOffset_Txphase).astype('int32')
dopplerBinsToSample = np.mod(dopplerBinsToSample, numDoppFFT)

DNL_rad = (DNL/180) * np.pi
noiseFloorSetByDNL = 10*np.log10((DNL_rad)**2/12) - 10*np.log10(numRamps) + 10*np.log10(numTx_simult) # DNL Noise floor raises as 10log10(numSimulTx)

powerMeanSpectrum_arossRxs = np.mean(signalFFTShiftSpectrum,axis=2) # Take mean spectrum across Rxs
noiseFloorEstFromSignal = 10*np.log10(np.percentile(powerMeanSpectrum_arossRxs,70,axis=1))
signalPowerDoppSpectrum = 10*np.log10(np.amax(powerMeanSpectrum_arossRxs,axis=1))
snrDoppSpectrum = signalPowerDoppSpectrum - noiseFloorEstFromSignal

""" Angle analysis"""
mimoCoefficients_eachDoppler_givenRange = signalFFT[:,dopplerBinsToSample,:] # numTx*numDopp x numRx
mimoCoefficients_flatten = mimoCoefficients_eachDoppler_givenRange.reshape(-1, numTx_simult*numRx)
mimoCoefficients_flatten = mimoCoefficients_flatten[:,ulaInd]
ULA = np.unwrap(np.angle(mimoCoefficients_flatten),axis=1).T
ULA -= ULA[0,:][None,:]

mimoCoeffWind = mimoCoefficients_flatten*np.hanning(numMIMO)[None,:]
ULA_spectrum = np.fft.fft(mimoCoeffWind,n=numAngleFFT,axis=1)/(numMIMO)
ULA_spectrum = np.fft.fftshift(ULA_spectrum,axes=(1,))
ULA_spectrumdB = 20*np.log10(np.abs(ULA_spectrum))
ULA_spectrumdB -= np.amax(ULA_spectrumdB,axis=1)[:,None]
ULA_spectrumdB = ULA_spectrumdB.T

estAngInd = np.argmax(ULA_spectrumdB,axis=0)
estAngDeg = angAxis_deg[estAngInd]
angleError = np.abs(estAngDeg - objectAzAngle_deg)

sllValdBc = np.zeros((numRangeOffsets),dtype=np.float32)
for ele in np.arange(numRangeOffsets):
    localMaxInd = argrelextrema(ULA_spectrumdB[:,ele],np.greater,axis=0,order=2)[0]
    sllInd = np.argsort(ULA_spectrumdB[localMaxInd,ele])[-2] # 1st SLL
    sllValdBc[ele] = ULA_spectrumdB[localMaxInd[sllInd],ele]


binshiftIndicesToPlot = np.arange(noRangeOffsetInd-3,noRangeOffsetInd+4)
binshiftsToPlot = rangebinOffset[binshiftIndicesToPlot]
legendLabel = ['True Rbin + ' + str(x) for x in binshiftsToPlot]

plt.figure(1, figsize=(20,10),dpi=150)
plt.subplot(1,2,1)
plt.title('Range spectrum')
plt.plot(10*np.log10(signal_rfft_powermean) + dBFs_to_dBm)
plt.xlabel('Range Bins')
plt.ylabel('Power dBm')
plt.grid(True)
plt.ylim([noiseFloor_perBin-10,0])

plt.subplot(1,2,2)
plt.title('Range bin movement across chirps. ' + 'Target Speed = ' + str(round(objectVelocity_mps)) + ' mps')
plt.plot(rangeBinsMoved[binshiftIndicesToPlot,:].T,'-o')
# plt.axhline(objectRangeBin,color='k',linestyle='dashed')
plt.xlabel('Chirp number')
plt.ylabel('Range bin')
plt.grid(True)
plt.legend(legendLabel)


plt.figure(3, figsize=(20,10), dpi=150)
plt.title('Doppler Spectrum with ' + str(numTx_simult) + 'Txs simultaneously ON in CDM. ' + 'Target Speed = ' + str(round(objectVelocity_mps)) + ' mps')
plt.plot(signalMagSpectrum[binshiftIndicesToPlot,:,0].T, lw=2) # Plotting only the 0th Rx instead of all 8
plt.xlabel('Doppler Bins')
plt.ylabel('Power dBFs')
plt.grid(True)
plt.legend(legendLabel)
plt.vlines(dopplerBinsToSample,ymin = np.amin(noiseFloorEstFromSignal)-20, ymax = np.amax(signalPowerDoppSpectrum)+5)


plt.figure(4, figsize=(20,10), dpi=150)
plt.subplot(1,2,1)
plt.title('MIMO phase')
plt.plot(ULA[:,binshiftIndicesToPlot],'-o')
plt.xlabel('Rx')
plt.ylabel('Phase (rad)')
plt.grid(True)
plt.legend(legendLabel)

plt.subplot(1,2,2)
plt.title('Angle spectrum')
plt.plot(angAxis_deg, ULA_spectrumdB[:,binshiftIndicesToPlot],label='Angle spectrum')
plt.xlabel('Angle (deg)')
plt.ylabel('dB')
plt.grid(True)
plt.legend(legendLabel)
plt.axvline(objectAzAngle_deg, color = 'k', label='Ground Truth angle (deg)')



plt.figure(5,figsize=(20,10),dpi=150)
plt.suptitle('DDMA: Angle sensitivity to bin offset')
plt.subplot(1,2,1)
plt.title('Angle accuracy vs Range bin offset')
plt.plot(rangebinOffset,angleError,'-o')
plt.xlabel('Range bin offset')
plt.ylabel('Angle error (deg)')
plt.grid(True)

plt.subplot(1,2,2)
plt.title('Angle SLLs vs Range bin offset')
plt.plot(rangebinOffset,sllValdBc,'-o')
plt.xlabel('Range bin offset')
plt.ylabel('dBc')
plt.grid(True)


