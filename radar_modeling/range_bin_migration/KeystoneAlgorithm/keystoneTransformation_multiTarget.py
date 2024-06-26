# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 00:16:44 2022

@author: saiguna
"""

"""
Reference paper used for Keystone transformation is available at the below link:
    https://saigunaranjan.atlassian.net/wiki/spaces/RM/pages/4227073/Keystone+Transformation
I have dervied and coded the keystone algorithm to cater to the FMCW architecture.
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
interRampTime = 44e-6#13e-6#44e-6 # us
rampSamplingRate = 1/interRampTime
chirpStartFreq = 77e9 # Giga Hz
numChirps = 168#500#168

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
numTargets = 3#int(np.random.randint(low=1, high=4, size=1))
objectRange_m = np.array([10,20,30]) #10 # np.array([10,20,30]) #10
objectVelocity_mps = np.array([0,10,-10]) #20#60 # 20 m/s np.array([0,20,-20])


rangeTerm = np.exp(1j*2*np.pi*(chirpSlope*2*objectRange_m[:,None]/lightSpeed)*\
                        adcSamplingTime*np.arange(numSamples)[None,:]) # numSamples

dopplerTerm = np.exp(1j*2*np.pi*(2*objectVelocity_mps[:,None]/lamda)*interRampTime*np.arange(numChirps)[None,:]) # DopplerBin x numChirps

rangeBinMigrationTerm = np.exp(1j*2*np.pi*chirpSlope*2*(objectVelocity_mps[:,None,None]/lightSpeed)*\
                                    interRampTime*adcSamplingTime*\
                                        np.arange(numChirps)[None,None,:]*np.arange(numSamples)[None,:,None]) # DopplerBin x numHyp x numChirps x numSamples

radarSignal = rangeTerm[:,:,None] * dopplerTerm[:,None,:] * rangeBinMigrationTerm
radarSignal = np.sum(radarSignal,axis=0) # sum across number of objects

receivedSignal = radarSignal
receivedSignal = receivedSignal*np.hanning(numSamples)[:,None]
receivedSignalRfft = np.fft.fft(receivedSignal,axis=0)/numSamples
receivedSignalRfft = receivedSignalRfft[0:numSamples//2,:]

receivedSignalRfftDoppWin = receivedSignalRfft*np.hanning(numChirps)[None,:]
receivedSignalRfftDoppFFT = np.fft.fft(receivedSignalRfftDoppWin,axis=1,n=numDoppFFT)/numChirps
receivedSignalRfftDoppFFT = np.fft.fftshift(receivedSignalRfftDoppFFT,axes=(1,))

receivedSignalRfftMagSpec = 20*np.log10(np.abs(receivedSignalRfft))
receivedSignalRfftDfftMagSpec = 20*np.log10(np.abs(receivedSignalRfftDoppFFT))

# rangeAxis = np.arange(numSamples//2)*rangeRes
chirpAxis = np.arange(numChirps)*interRampTime


""" KeyStone Transformation"""
interpFact = (chirpCentreFreq/(chirpCentreFreq + chirpSlope*np.arange(numSamples)[:,None]*adcSamplingTime))*np.arange(numChirps)[None,:]*interRampTime

allChirpsDuration = (numChirps-1)*interRampTime

doppPhase = np.unwrap(np.angle(radarSignal),axis=1)
doppSignal = radarSignal
chirpSampPoints = np.arange(numChirps)*interRampTime
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

        # doppPhaseInterpFunc = interpolate.splrep(chirpSampPoints, doppPhase[ele,:], k=3) # cubic spline interpolation
        # xnew = interpFact[ele,:]
        # doppPhaseInterpVals = interpolate.splev(xnew, doppPhaseInterpFunc, der=0)

        doppSignalRealInterpFunc = interpolate.splrep(chirpSampPoints, np.real(doppSignal[ele,:]), k=5) # cubic spline interpolation
        doppSignalImagInterpFunc = interpolate.splrep(chirpSampPoints, np.imag(doppSignal[ele,:]), k=5) # cubic spline interpolation
        xnew = interpFact[ele,:]
        doppSignalRealInterpVals = interpolate.splev(xnew, doppSignalRealInterpFunc, der=0)
        doppSignalImagInterpVals = interpolate.splev(xnew, doppSignalImagInterpFunc, der=0)

    # interpreceivedSignal[ele,:] = np.exp(1j*doppPhaseInterpVals)
    interpreceivedSignal[ele,:] = doppSignalRealInterpVals + 1j*doppSignalImagInterpVals


# doppPhaseInterp = np.unwrap(np.angle(interpreceivedSignal),axis=1)
interpreceivedSignalWind = interpreceivedSignal*np.hanning(numSamples)[:,None]
interpreceivedSignalRfft = np.fft.fft(interpreceivedSignalWind,axis=0)/numSamples
interpreceivedSignalRfft = interpreceivedSignalRfft[0:numSamples//2,:]

interpreceivedSignalRfftDoppWin = interpreceivedSignalRfft*np.hanning(numChirps)[None,:]
interpreceivedSignalRfftDoppFFT = np.fft.fft(interpreceivedSignalRfftDoppWin,axis=1,n=numDoppFFT)/numChirps
interpreceivedSignalRfftDoppFFT = np.fft.fftshift(interpreceivedSignalRfftDoppFFT,axes=(1,))

interpreceivedSignalRfftMagSpec = 20*np.log10(np.abs(interpreceivedSignalRfft))
interpreceivedSignalRfftDfftMagSpec = 20*np.log10(np.abs(interpreceivedSignalRfftDoppFFT))

binsMovedPreCorr = np.argmax(receivedSignalRfftMagSpec,axis=0)
binsMovedPostCorr = np.argmax(interpreceivedSignalRfftMagSpec,axis=0)

plt.figure(1,figsize=(20,10),dpi=200)
plt.suptitle('Range spectrum')
plt.subplot(1,2,1)
plt.title('Before keystone correction')
plt.plot(receivedSignalRfftMagSpec)
plt.xlabel('Range bins')
plt.grid(True)

plt.subplot(1,2,2)
plt.title('Post keystone correction')
plt.plot(interpreceivedSignalRfftMagSpec)
plt.xlabel('Range bins')
plt.grid(True)


plt.figure(2,figsize=(20,10),dpi=200)
plt.subplot(2,2,1)
plt.title('Range bin vs Chirp number')
plt.imshow(receivedSignalRfftMagSpec,aspect='auto')
plt.xlabel('Chirp number')
plt.ylabel('Range bin')
plt.subplot(2,2,2)
plt.title('Range-Doppler map')
plt.imshow(receivedSignalRfftDfftMagSpec,vmin=-50,vmax=-10,cmap='afmhot')
plt.colorbar()
plt.xlabel('Doppler bin')
plt.ylabel('Range bin')
plt.subplot(2,2,3)
plt.imshow(interpreceivedSignalRfftMagSpec,aspect='auto')
plt.xlabel('Chirp number')
plt.ylabel('Range bin')
plt.subplot(2,2,4)
plt.imshow(interpreceivedSignalRfftDfftMagSpec,vmin=-50,vmax=-10,cmap='afmhot')
plt.colorbar()
plt.xlabel('Doppler bin')
plt.ylabel('Range bin')

# plt.figure(3,figsize=(20,10),dpi=200)
# plt.title('Range bins moved across chirps for object with speed = ' + str(objectVelocity_mps) + ' mps.' + ' Chirp BW = ' + str(int(chirpBW/1e9)) + ' GHz')
# plt.plot(binsMovedPreCorr,'-o',label='Pre correction')
# plt.plot(binsMovedPostCorr,'-o',label='Post Keystone transformation')
# plt.xlabel('chirp number')
# plt.ylabel('range bins');
# plt.grid(True)
# plt.legend()
