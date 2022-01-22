# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 23:04:53 2021

@author: saiguna
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
# np.random.seed(0)

""" System Parameters """

""" Below are the system parameters used for the 256v2J platform"""
numSamples = 2048
numFFTBins = 2048 #8192
thermalNoise = -174 # dBm/Hz
noiseFigure = 10 # dB
baseBandgain = 34 #dB
adcSamplingRate = 56.25e6 # 56.25 MHz
dBFs_to_dBm = 10
binSNR = 20 # dB
lightSpeed = 3e8 # m/s

""" Chirp Parameters"""
chirpBW = 4e9 # Hz
interRampTime = 44e-6 # us

chirpStartFreq = 77e9 # Giga Hz
numChirps = 168


""" Object parameters"""
objectRange_m = 10
objectVelocity_mps = 40 # m/s



fftOverSamplingFact = numFFTBins//numSamples
adcSamplingTime = 1/adcSamplingRate # seconds
chirpOnTime = numSamples*adcSamplingTime #39.2e-6

totalNoisePower_dBm = thermalNoise + noiseFigure + baseBandgain + 10*np.log10(adcSamplingRate)
totalNoisePower_dBFs = totalNoisePower_dBm - 10
noiseFloor_perBin = totalNoisePower_dBFs - 10*np.log10(numFFTBins) # dBFs/bin
# noiseFloor_perBin = -100 # dBFs
noisePower_perBin = 10**(noiseFloor_perBin/10)


totalNoisePower = noisePower_perBin*numFFTBins # sigmasquare totalNoisePower
sigma = np.sqrt(totalNoisePower)
signalPowerdBFs = noiseFloor_perBin + binSNR
signalPower = 10**(signalPowerdBFs/10)
signalAmplitude = np.sqrt(signalPower)
signalPhase = np.exp(1j*np.random.uniform(-np.pi, np.pi))
signalphasor = signalAmplitude*signalPhase

rangeRes = lightSpeed/(2*chirpBW)
rangeAxis_m = np.arange(numFFTBins//2)*rangeRes
objectRangeBin = objectRange_m/rangeRes
objectRangeBinInt = np.int(objectRangeBin)
rangeSignal = signalphasor*np.exp(1j*((2*np.pi*objectRangeBin)/numSamples)*np.arange(numSamples))


chirpSamplingRate = 1/interRampTime
chirpSlope = chirpBW/chirpOnTime
chirpCentreFreq = chirpStartFreq + chirpBW/2
lamda = lightSpeed/chirpCentreFreq
maxVelBaseband_mps = (chirpSamplingRate/2) * (lamda/2) # m/s
print('Max base band velocity = {0:.2f} m/s'.format(maxVelBaseband_mps))
FsEquivalentVelocity = 2*maxVelBaseband_mps # Fs = 2*Fs/2
velocityRes = (chirpSamplingRate/numChirps) * (lamda/2)
print('Velocity resolution = {0:.2f} m/s'.format(velocityRes))

objectVelocity_baseBand_mps = np.mod(objectVelocity_mps, FsEquivalentVelocity) # modulo Fs [from 0 to Fs]
objectVelocityBin = objectVelocity_baseBand_mps/velocityRes
objectVelocityInt = np.int(objectVelocityBin)

dopplerSignal = np.exp(1j*((2*np.pi*objectVelocityBin)/numChirps)*np.arange(numChirps))

""" Range Bin migration term"""
rangeBinMigration = \
    np.exp(1j*2*np.pi*chirpSlope*(2*objectVelocity_mps/lightSpeed)*interRampTime*adcSamplingTime*np.arange(numSamples)[:,None]*np.arange(numChirps)[None,:])

radarSignal = rangeSignal[:,None] * dopplerSignal[None,:] * rangeBinMigration

noise = (sigma/np.sqrt(2))*np.random.randn(numSamples*numChirps) + 1j*(sigma/np.sqrt(2))*np.random.randn(numSamples*numChirps)
noise = noise.reshape(numSamples, numChirps)
signal_noise = radarSignal + noise

receivedSignal = signal_noise*np.hanning(numSamples)[:,None]

rangeFFTSignal = np.fft.fft(receivedSignal, n=numFFTBins, axis=0)/numSamples
rangeFFTSignal = rangeFFTSignal[0:numFFTBins//2,:]


# spectrum = np.abs(rangeFFTSignal)**2
# estSignalPower = 10*np.log10(np.sum(spectrum[fftOverSamplingFact*(objectRangeBinInt-3):fftOverSamplingFact*(objectRangeBinInt+4)]))
# estNoiseFloor = 10*np.log10(np.mean(np.sort(spectrum)[0:numFFTBins//2]))


# print('Estimated Signal Power = {} dBFs'.format(np.round(estSignalPower)))
# print('True Noise power/bin = {} dBFs'.format(np.round(noiseFloor_perBin)))
# print('Estimated Noise power/bin = {} dBFs'.format(np.round(estNoiseFloor)))
# print('Estimated SNR = {}'.format(np.round(estSignalPower-estNoiseFloor)))

signalSpectrumdBm = 20*np.log10(np.abs(rangeFFTSignal)) + dBFs_to_dBm

plt.figure(1,figsize=(20,10))
plt.plot(rangeAxis_m, signalSpectrumdBm)
plt.axvline(objectRange_m, color = 'k')
plt.xlabel('Range (m)')
plt.ylabel('dBm')
plt.grid(True)




"""
Matched Filter construction

rangeTerm = exp(1j*2*pi*S*2*d/c * Ts*n)
dopplerTerm = exp(1j*2*pi*fc*2v/c * Tc*k)
rangeMigrationTerm = exp(1j*2*pi*S*2v/c * Tc*Ts*k*n)

"""
rangeBins_LUT = np.arange(numSamples)
distanceArray = rangeBins_LUT*rangeRes
selDistBin = numSamples//4
selDist =  selDistBin * rangeRes

rangeTermModel = np.exp(1j*2*np.pi*(chirpSlope*2*selDist/lightSpeed)*\
                        adcSamplingTime*np.arange(numSamples)) # numSamples
# rangeTermModel = rangeTermModel.astype('complex64')

dopplerBin_bipolar = np.arange(numChirps) - numChirps//2
velocityArray_baseband = dopplerBin_bipolar[:,None]*velocityRes
doppIntHyp = np.arange(-1,2)
velocityArray = velocityArray_baseband + doppIntHyp[None,:]*FsEquivalentVelocity
dopplerTermModel = np.exp(1j*2*np.pi*(2*velocityArray_baseband/lamda)*interRampTime*np.arange(numChirps)[None,:]) # DopplerBin x numChirps
# dopplerTermModel = dopplerTermModel.astype('complex64')

rangeBinMigrationTermModel = np.exp(1j*2*np.pi*chirpSlope*2*(velocityArray[:,:,None,None]/lightSpeed)*\
                                    interRampTime*adcSamplingTime*\
                                        np.arange(numChirps)[None,None,:,None]*np.arange(numSamples)[None,None,None,:]) # DopplerBin x numHyp x numChirps x numSamples
# rangeBinMigrationTermModel = rangeBinMigrationTermModel.astype('complex64')

rangeTerm_binMigrationTerm = rangeTermModel[None,None,None,:] * rangeBinMigrationTermModel # [DoppplerBin, Hyp, numChirps, numSamples]
rangeTerm_binMigrationTerm_Windowed = np.hanning(numSamples)[None,None,None,:]*rangeTerm_binMigrationTerm
range_binMigration_fft = np.fft.fft(rangeTerm_binMigrationTerm_Windowed,axis=3)/numSamples
range_binMigration_fft = range_binMigration_fft[:,:,:,0:numSamples//2]

maxBinMigration = np.ceil(((maxVelBaseband_mps + doppIntHyp[-1]*FsEquivalentVelocity) * numChirps * interRampTime)/rangeRes).astype('int32')
range_binMigration_fft_pruned = range_binMigration_fft[:,:,:,selDistBin-maxBinMigration:selDistBin+maxBinMigration+1]
matchedFilter = range_binMigration_fft_pruned * dopplerTermModel[:,None,:,None]


""" Apply Matched filter on the received range FFT signal to estimte the true aliased velocity"""
rangeBintoDial = objectRangeBinInt
rangeBinWindow = np.arange(rangeBintoDial-maxBinMigration, rangeBintoDial+maxBinMigration+1)

signalofInterest = rangeFFTSignal[rangeBinWindow,:].T
correlationEnergy = np.abs(np.sum(matchedFilter * np.conj(signalofInterest)[None,None,:,:],axis=(2,3)))
flatInd = np.argmax(correlationEnergy)
estDoppInd, estDoppHypInd = np.unravel_index(flatInd,(correlationEnergy.shape[0],correlationEnergy.shape[1]))

estTrueVelocity = velocityArray_baseband[estDoppInd] + doppIntHyp[estDoppHypInd]*FsEquivalentVelocity

print('True Velocity = {} m/s'.format(objectVelocity_mps))
print('Estimated Velocity = {0:.2f} m/s'.format(estTrueVelocity[0]))






