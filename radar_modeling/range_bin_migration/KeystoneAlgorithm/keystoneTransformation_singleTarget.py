# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 00:16:44 2022

@author: saiguna
"""

"""
Reference paper used for Keystone transformation is available at the below link:
    https://saigunaranjan.atlassian.net/wiki/spaces/RM/pages/4227073/Keystone+Transformation
I have derived and coded the keystone algorithm to cater to the FMCW architecture. The derivation is available in the below link:
https://saigunaranjan.atlassian.net/wiki/spaces/RM/pages/4227073/Keystone+Transformation+for+FMCW+RADAR+architecture+My+derivation

This script caters to the Keystone transformation for a single Range Doppler target
with any Doppler(aliased/unaliased).

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import interp1d



plt.close('all')
# np.random.seed(0)

""" System Parameters """

""" Below are the system parameters used for the 256v2J platform"""
numSamples = 2048
numFFTBins = 2048
numDoppFFT = 2048
thermalNoise = -174 # dBm/Hz
noiseFigure = 10 # dB
baseBandgain = 34 #dB
adcSamplingRate = 56.25e6 # MHz
dBFs_to_dBm = 10
binSNR = 20 # dB
lightSpeed = 3e8 # m/s

""" Chirp Parameters"""
chirpBW = 4e9 # Hz
interRampTime = 44e-6 # us
rampSamplingRate = 1/interRampTime
chirpStartFreq = 77e9 # Giga Hz
numChirps = 168

""" Derived Parameters"""
adcSamplingTime = 1/adcSamplingRate # seconds
chirpOnTime = numSamples*adcSamplingTime #39.2e-6
chirpSamplingRate = 1/interRampTime
chirpSlope = chirpBW/chirpOnTime
chirpCentreFreq = chirpStartFreq + chirpBW/2
lamda = lightSpeed/chirpCentreFreq
maxVelBaseband_mps = (chirpSamplingRate/2) * (lamda/2) # m/s
FsEquivalentVelocity = 2*maxVelBaseband_mps # Fs = 2*Fs/2
velocityRes = (chirpSamplingRate/numChirps) * (lamda/2)
rangeRes = lightSpeed/(2*chirpBW)
maxRange = (numSamples//2)*rangeRes

print('Max Range = {0:.2f} m'.format(maxRange))
print('Range resolution = {0:.2f} cm'.format(rangeRes*100))
print('Max base band velocity = {0:.2f} m/s'.format(maxVelBaseband_mps))
print('Velocity resolution = {0:.2f} m/s'.format(velocityRes))


""" Object parameters"""
objectRange_m = 10
doppHyp = np.arange(-2,3) #np.array([1])
basebandVelocity = np.random.uniform(-maxVelBaseband_mps+2, maxVelBaseband_mps-2) #15
selectDoppIntHyp = np.random.randint(doppHyp[0], doppHyp[-1]+1)#1 # low value is inclusive and high value is exclusive and hence +1
objectVelocity_mps = basebandVelocity + selectDoppIntHyp*FsEquivalentVelocity

rangeTerm = np.exp(1j*2*np.pi*(chirpSlope*2*objectRange_m/lightSpeed)*\
                        adcSamplingTime*np.arange(numSamples)) # numSamples

dopplerTerm = np.exp(1j*2*np.pi*(2*objectVelocity_mps/lamda)*interRampTime*np.arange(numChirps)) # DopplerBin x numChirps

rangeBinMigrationTerm = np.exp(1j*2*np.pi*chirpSlope*2*(objectVelocity_mps/lightSpeed)*\
                                    interRampTime*adcSamplingTime*\
                                        np.arange(numChirps)[None,:]*np.arange(numSamples)[:,None]) # DopplerBin x numHyp x numChirps x numSamples

radarSignal = rangeTerm[:,None] * dopplerTerm[None,:] * rangeBinMigrationTerm

receivedSignal = radarSignal
receivedSignal = receivedSignal*np.hanning(numSamples)[:,None]
receivedSignalRfft = np.fft.fft(receivedSignal,axis=0)/numSamples
receivedSignalRfft = receivedSignalRfft[0:numSamples//2,:]

receivedSignalRfftDoppWin = receivedSignalRfft*np.hanning(numChirps)[None,:]
receivedSignalRfftDoppFFT = np.fft.fft(receivedSignalRfftDoppWin,axis=1,n=numDoppFFT)/numChirps
receivedSignalRfftDoppFFT = np.fft.fftshift(receivedSignalRfftDoppFFT,axes=(1,))

receivedSignalRfftMagSpec = 20*np.log10(np.abs(receivedSignalRfft))
receivedSignalRfftDfftMagSpec = 20*np.log10(np.abs(receivedSignalRfftDoppFFT))


chirpAxis = np.arange(numChirps)*interRampTime


""" KeyStone Transformation"""
interpFact = (chirpCentreFreq/(chirpCentreFreq + chirpSlope*np.arange(numSamples)[:,None]*adcSamplingTime))*np.arange(numChirps)[None,:]*interRampTime

doppPhase = np.unwrap(np.angle(radarSignal),axis=1)
chirpSampPoints = np.arange(numChirps)*interRampTime
""" interpolated and resampled signal.
interpreceivedSignal satnds for interpolated and resampled received signal"""
interpreceivedSignal = np.zeros((receivedSignal.shape),dtype=np.complex64)
for ele in range(numSamples):
    if 0:
        """ 1-D linear interpolation. Valid only when there is a single Doppler in the scene"""
        doppPhaseInterpFunc = interp1d(chirpSampPoints, doppPhase[ele,:], kind='linear')
        xnew = interpFact[ele,:]
        doppPhaseInterpVals = doppPhaseInterpFunc(xnew)
    if 1:
        """ Spline interpolation. Valid even when there are multiple Dopplers in the scene"""
        # doppPhaseInterpFunc = interpolate.splrep(chirpSampPoints, doppPhase[ele,:], s=0, k=1)
        doppPhaseInterpFunc = interpolate.splrep(chirpSampPoints, doppPhase[ele,:], k=3) # cubic spline interpolation
        xnew = interpFact[ele,:]
        doppPhaseInterpVals = interpolate.splev(xnew, doppPhaseInterpFunc, der=0)

    interpreceivedSignal[ele,:] = np.exp(1j*doppPhaseInterpVals)

""" Doppler Ambiguity Correction Term"""
DoppHypCorrFactor = np.exp(+1j*2*np.pi*((chirpCentreFreq/(chirpCentreFreq + chirpSlope*np.arange(numSamples)[:,None,None]*adcSamplingTime))*np.arange(numChirps)[None,:,None]*doppHyp[None,None,:]))

""" Apply Doppler Ambiguity Correction Term to the Keystone transformation based interpolated and resampled signal"""
interpreceivedSignalDoppHypCorrected = interpreceivedSignal[:,:,None]*DoppHypCorrFactor # [numADCSamp, numRamps, numDoppHyp]

interpreceivedSignalWind = interpreceivedSignalDoppHypCorrected*np.hanning(numSamples)[:,None,None]
interpreceivedSignalRfft = np.fft.fft(interpreceivedSignalWind,axis=0)/numSamples
interpreceivedSignalRfft = interpreceivedSignalRfft[0:numSamples//2,:,:]
interpreceivedSignalRfftDoppWin = interpreceivedSignalRfft*np.hanning(numChirps)[None,:,None]
interpreceivedSignalRfftDoppFFT = np.fft.fft(interpreceivedSignalRfftDoppWin,axis=1,n=numDoppFFT)/numChirps

targetRbinToSamp = np.round(objectRange_m/rangeRes).astype(np.int32)
targetDbinToSamp = np.round(((basebandVelocity/velocityRes)/numChirps)*numDoppFFT).astype(np.int32)
powerSpectrumVals = np.abs(interpreceivedSignalRfftDoppFFT[targetRbinToSamp,targetDbinToSamp,:])
""" Search for the Doppler Integer Hypothesis which gives the maximum energy """
DoppHypMaxInd = np.argmax(powerSpectrumVals)
estDoppHyp = doppHyp[DoppHypMaxInd]

estTrueVel = basebandVelocity + estDoppHyp*FsEquivalentVelocity
print('\n True Velocity = {0:.2f} m/s'.format(objectVelocity_mps))
print('Estimated True Velocity = {0:.2f} m/s'.format(estTrueVel))

""" Sampling the range Doppler signal corresponding to the correct Doppler Ambiguity number/Hyp"""
interpreceivedSignalRfftDoppFFT = np.fft.fftshift(interpreceivedSignalRfftDoppFFT[:,:,DoppHypMaxInd],axes=(1,))
interpreceivedSignalRfftMagSpec = 20*np.log10(np.abs(interpreceivedSignalRfft[:,:,DoppHypMaxInd]))
interpreceivedSignalRfftDfftMagSpec = 20*np.log10(np.abs(interpreceivedSignalRfftDoppFFT))

binsMovedPreCorr = np.argmax(receivedSignalRfftMagSpec,axis=0)
totalBinsMoved = np.abs(binsMovedPreCorr[-1] - binsMovedPreCorr[0])
binsMovedPostCorr = np.argmax(interpreceivedSignalRfftMagSpec,axis=0)

doppResFFTScale = (chirpSamplingRate/numDoppFFT) * (lamda/2)
dopplerAxis = np.arange(-numDoppFFT//2, numDoppFFT//2)*doppResFFTScale


plt.figure(1,figsize=(20,10),dpi=200)
plt.suptitle('Target speed = ' + str(np.round(objectVelocity_mps,2)) + ' mps.' + ' Chirp BW = ' + str(int(chirpBW/1e9)) + ' GHz')
plt.subplot(2,2,1)
plt.title('Range bin vs Chirp number')
plt.imshow(receivedSignalRfftMagSpec[targetRbinToSamp-totalBinsMoved:targetRbinToSamp+totalBinsMoved,:])
plt.xlabel('Chirp number')
plt.ylabel('Range bin')
plt.subplot(2,2,2)
plt.title('Range-Doppler map')
plt.imshow(receivedSignalRfftDfftMagSpec,vmin=-110,vmax=0,cmap='afmhot')
plt.colorbar()
plt.xlabel('Doppler bin')
plt.ylabel('Range bin')
plt.subplot(2,2,3)
plt.imshow(interpreceivedSignalRfftMagSpec[targetRbinToSamp-totalBinsMoved:targetRbinToSamp+totalBinsMoved,:])
plt.xlabel('Chirp number')
plt.ylabel('Range bin')
plt.subplot(2,2,4)
plt.imshow(interpreceivedSignalRfftDfftMagSpec,vmin=-110,vmax=0,cmap='afmhot')
plt.colorbar()
plt.xlabel('Doppler bin')
plt.ylabel('Range bin')

plt.figure(2,figsize=(20,10),dpi=200)
plt.title('Range bins moved across chirps for object with speed = ' + str(np.round(objectVelocity_mps,2)) + ' mps.' + ' Chirp BW = ' + str(int(chirpBW/1e9)) + ' GHz')
plt.plot(binsMovedPreCorr,'-o',label='Pre correction')
plt.plot(binsMovedPostCorr,'-o',label='Post Keystone correction')
plt.xlabel('chirp number')
plt.ylabel('range bins');
plt.grid(True)
plt.legend()

plt.figure(3,figsize=(20,10),dpi=200)
plt.suptitle('Doppler Spectrum: ' + str(np.round(objectVelocity_mps,2)) + \
             ' mps velocity folded back to ' + str(np.round(basebandVelocity,2)) + ' mps')
plt.plot(dopplerAxis, receivedSignalRfftDfftMagSpec[binsMovedPreCorr[0],:], label='Before Keystone correction')
plt.plot(dopplerAxis, interpreceivedSignalRfftDfftMagSpec[binsMovedPostCorr[0],:],label='After Keystone correction')
plt.axvline(basebandVelocity,color='k',label='Ground truth velocity')
plt.xlabel('Velocity (mps)')
plt.ylim([-174,-5])
plt.grid('True')
plt.legend()


plt.figure(4,figsize=(20,10),dpi=200)
plt.suptitle('Energy(dB) vs Dopp Ambiguity number')
plt.bar(doppHyp, 20*np.log10(powerSpectrumVals),label='Energy')
plt.axvline(selectDoppIntHyp,color='k',label='Ground truth')
plt.xlabel('Doppler Ambiguity Number')
plt.ylabel('dB')
plt.legend()
plt.grid('True')






