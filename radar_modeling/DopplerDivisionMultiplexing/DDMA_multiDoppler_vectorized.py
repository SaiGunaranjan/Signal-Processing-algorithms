# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:36:29 2021

@author: saiguna
"""

""" In this script, I have modeled how the phase shifter error(from the
ground truth)for each unique phase shifter manifests as periodic repetetions
when the phase is swept over several ramps. This periodic component for each
of the phase shifter manifests as spurs in the Doppler spectrum. These spurs get
aggravated when the phase delta(in degrees) programmed per ramp divides 360 degress
exactly. In such case, the same phase shifter code is applied periodically
at every 360/phase_delta ramp number. For example, if the phase delta per ramp
is chosen as 30 degrees, then 360/30 = 12 i.e. the starting ramp and the 12th ramp
will have the same phase shifter applied. Similarly, the 2nd ramp and the 13th ramp
have the same phase shifter applied and so on. Now since each phase shifter
has its own "signature"(error in terms of DNL wrt ground truth), there will be
a periodic pattern every 12th ramp. To break this periodicity, I have chosen
a phase step which doesn't divide 360 degrees. For example, if the phase step per ramp
is 29 degress instead of 30 degress, then at the 13th ramp, the phase is 29*13 = 377 = 17 degress.
But this is not equal to 0 degrees i.e the phase shifter for the 13th ramp
is not the same as the phase shifter for the 1st ramp. Similarly, the second ramp
has a phase of 29 degress while the 14th ramp has a phase of 29*14= 406 = 46 degress which
is not the same as 29 degress. Hence the phase shifter for the 14th ramp
is not the same as the phase shifter for the 2nd ramp. Hence just by making
the phase change per ramp not a divisor of 360 degress, we remove the periodicity.
We also observe that the noise floor increases by 3 dB for each Tx that is added
to the simultaneous transmission i.e., when we move from 3 Tx to 4 Tx, the noise floor raises by another 3 dB.
This is not very clear to me and I need to understand this better!!
"""

""" In this commit, I have also modeled the Txs/Rxs with simulataneous transmission from all Txs
each with its own phase code per ramp and have also been able to estimate MIMO coeficients.
I have modelled the DDMA for the Steradian SRIR144 and SRIR256 platforms.
"""

""" The derivation for the DDMA scheme is available in the below location:
    https://saigunaranjan.atlassian.net/wiki/spaces/RM/pages/1966081/Code+Division+Multiple+Access+in+FMCW+RADAR"""

import numpy as np
import matplotlib.pyplot as plt
from mimoPhasorSynthesis import mimoPhasorSynth # loading the antenna corodinates for the steradian RADAR platforms


plt.close('all')

numDopUniqRbin = int(np.random.randint(low=1, high=4, size=1)) #2 # Number of Dopplers in a given range bin
flagRBM = 1
if (flagRBM == 1):
    print('\n\nRange Bin Migration term has been enabled\n\n')

platform = 'SRIR16' # 'SRIR16', 'SRIR256', 'SRIR144'

if (platform == 'SRIR16'):
    numTx_simult = 4
    numRx = 4
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
c = 3e8
numBitsPhaseShifter = 7
numPhaseCodes = 2**numBitsPhaseShifter
DNL = 360/(numPhaseCodes) # DNL in degrees

""" Chirp Parameters"""
numDoppFFT = 2048
chirpBW = 1e9 # Hz
centerFreq = 76.5e9 # GHz
interRampTime = 44e-6 # us
rampSamplingRate = 1/interRampTime
rangeRes = c/(2*chirpBW)
maxRange = numSampPostRfft*rangeRes # m
lamda = lightSpeed/centerFreq
""" With 30 deg, we see periodicity since 30 divides 360 but with say 29 deg, it doesn't divide 360 and hence periodicity is significantly reduced"""
phaseStepPerTx_deg = 29#29.3

Fs_spatial = lamda/mimoArraySpacing
angAxis_deg = np.arcsin(np.arange(-numAngleFFT//2, numAngleFFT//2)*(Fs_spatial/numAngleFFT))*180/np.pi


""" Derived Parameters """
snrGainDDMA = 10*np.log10(numTx_simult**2) #dB
snrGainDopplerFFT = 10*np.log10(numRamps) #dB
totalsnrGain = snrGainDDMA + snrGainDopplerFFT
print('Total SNR gain ( {0:.0f} Tx DDMA + {1:.0f} point Doppler FFT) = {2:.2f} dB'.format(numTx_simult, numRamps, totalsnrGain))

chirpSamplingRate = 1/interRampTime
maxVelBaseband_mps = (chirpSamplingRate/2) * (lamda/2) # m/s
print('Max base band velocity = {0:.2f} m/s'.format(maxVelBaseband_mps))
FsEquivalentVelocity = 2*maxVelBaseband_mps # Fs = 2*Fs/2
velocityRes = (chirpSamplingRate/numRamps) * (lamda/2)
print('Velocity resolution = {0:.2f} m/s'.format(velocityRes))

""" Target definition"""
objectRange = np.random.uniform(10,maxRange-10) # 60.3 # m
objectVelocity_mps = np.random.uniform(-maxVelBaseband_mps-2*FsEquivalentVelocity, \
                                        maxVelBaseband_mps+2*FsEquivalentVelocity, numDopUniqRbin)  #np.array([-10,-10.1]) #np.array([-10,23])# m/s

print('Velocities (mps):', np.round(objectVelocity_mps,2))
objectAzAngle_deg = np.random.uniform(-50,50, numDopUniqRbin) #np.array([30,-10]) Theta plane angle
objectAzAngle_rad = (objectAzAngle_deg/360) * (2*np.pi)

objectElAngle_deg = np.zeros((numDopUniqRbin,)) # phi=0 plane angle
objectElAngle_rad = (objectElAngle_deg/360) * (2*np.pi)

mimoPhasor, mimoPhasor_txrx, ulaInd = mimoPhasorSynth(platform, lamda, objectAzAngle_rad, objectElAngle_rad)


## RF parameters
thermalNoise = -174 # dBm/Hz
noiseFigure = 10 # dB
baseBandgain = 34 #dB
adcSamplingRate = 56.25e6 # 56.25 MHz
adcSamplingTime = 1/adcSamplingRate # s
chirpOnTime = numSamp*adcSamplingTime
chirpSlope = chirpBW/chirpOnTime
dBFs_to_dBm = 10
binSNR = -3 # dB
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


objectVelocity_baseBand_mps = np.mod(objectVelocity_mps, FsEquivalentVelocity) # modulo Fs [from 0 to Fs]
objectVelocityBin = objectVelocity_baseBand_mps/velocityRes
objectRangeBin = objectRange/rangeRes
if (flagRBM == 1):
    rangeMoved = objectRange + objectVelocity_mps[:,None]*interRampTime*np.arange(numRamps)[None,:]
else:
    rangeMoved = objectRange + 0*objectVelocity_mps[:,None]*interRampTime*np.arange(numRamps)[None,:]

rangeBinsMoved = np.floor(rangeMoved/rangeRes).astype('int32')



phaseStepPerRamp_deg = np.arange(numTx_simult)*phaseStepPerTx_deg # Phase step per ramp per Tx
phaseStepPerRamp_rad = (phaseStepPerRamp_deg/360)*2*np.pi

phaseShifterCodes = DNL*np.arange(numPhaseCodes)
phaseShifterNoise = np.random.uniform(-DNL/2, DNL/2, numPhaseCodes)
phaseShifterCodes_withNoise = phaseShifterCodes + phaseShifterNoise
""" Ensure that the phase shifter LUT is without any bias and is from 0 to 360 degrees"""
phaseShifterCodes_withNoise = np.mod(phaseShifterCodes_withNoise,360)

""" NOT USING THE BELOW QUADRATIC PHASE in this DDMA MIMO scheme

A small quadratic term is added to the linear phase term to break the periodicity.
The strength of the quadratic term is controlled by the alpha parameter. If alpha is large,
the quadratic term dominates the linear term thus breaking the periodicity of the Phase shifter DNL but
at the cost of main lobe widening.
If alpha is very small, the contribution of the quadratic term diminishes and the periodicity of
the phase shifter DNL is back thus resulting in spurs/harmonics in the spectrum.
Thus the alpha should be moderate to break the periodicity with help
of the quadratic term at the same time not degrade the main lobe width.
I have observed that for phaseStepPerRamp_deg=30 deg, alpha = 1e-4 is best. We will use this
to scale the alpha for other step sizes"""

alpha = 0#1e-4
rampPhaseIdeal_deg = phaseStepPerRamp_deg[:,None]*(np.arange(numRamps)[None,:] + (alpha/2)*(np.arange(numRamps)[None,:])**2)
rampPhaseIdeal_degWrapped = np.mod(rampPhaseIdeal_deg, 360)
phaseCodesIndexToBeApplied = np.argmin(np.abs(rampPhaseIdeal_degWrapped[:,:,None] - phaseShifterCodes_withNoise[None,None,:]),axis=2)
phaseCodesToBeApplied = phaseShifterCodes_withNoise[phaseCodesIndexToBeApplied]
phaseCodesToBeApplied_rad = (phaseCodesToBeApplied/180) * np.pi

rangeTerm = signalphasor*np.exp(1j*((2*np.pi*objectRangeBin)/numSamp)*np.arange(numSamp))
dopplerTerm = np.exp(1j*((2*np.pi*objectVelocityBin[:,None])/numRamps)*np.arange(numRamps)[None,:]) # [number of Dopplers/range, numRamps]
""" Range Bin migration term"""
rangeBinMigration = \
    np.exp(1j*2*np.pi*chirpSlope*(2*objectVelocity_mps[:,None,None]/lightSpeed)*interRampTime*adcSamplingTime*np.arange(numRamps)[None,:,None]*np.arange(numSamp)[None,None, :])

# rxSignal = np.exp(1j*(2*np.pi/lamda)*rxSpacing*np.sin(objectAzAngle_rad[:,None])*np.arange(numRx)[None,:]) # [number of Angles/RD, numRx]
# txSignal = np.exp(1j*(2*np.pi/lamda)*txSpacing*np.sin(objectAzAngle_rad[:,None])*np.arange(numTx_simult)[None,:]) # [number of Angles/RD, numTx]

rxSignal = mimoPhasor_txrx[:,0,:]
txSignal = mimoPhasor_txrx[:,:,0]

signal_phaseCode = np.exp(1j*phaseCodesToBeApplied_rad)
phaseCodedTxSignal = dopplerTerm[:,None,:] * signal_phaseCode[None,:,:] * txSignal[:,:,None] # [numDopp, numTx, numRamps]
phaseCodedTxRxSignal = phaseCodedTxSignal[:,:,:,None]*rxSignal[:,None,None,:] #[numDopp, numTx, numRamps, numTx, numRx]
phaseCodedTxRxSignal_withRangeTerm = rangeTerm[None,None,None,None,:] * phaseCodedTxRxSignal[:,:,:,:,None]
if (flagRBM == 1):
    phaseCodedTxRxSignal_withRangeTerm = phaseCodedTxRxSignal_withRangeTerm * rangeBinMigration[:,None,:,None,:]
signal = np.sum(phaseCodedTxRxSignal_withRangeTerm, axis=(0,1)) # [numRamps,numRx, numSamp]


noise = (sigma/np.sqrt(2))*np.random.randn(numRamps*numRx*numSamp) + 1j*(sigma/np.sqrt(2))*np.random.randn(numRamps*numRx*numSamp)
noise = noise.reshape(numRamps, numRx, numSamp)
signal = signal + noise

signal_rangeWin = signal*np.hanning(numSamp)[None,None,:]
signal_rfft = np.fft.fft(signal_rangeWin,axis=2)/numSamp
signal_rfft = signal_rfft[:,:,0:numSampPostRfft]
signal_rfft_powermean = np.mean(np.abs(signal_rfft)**2,axis=(0,1))

rangeBinsToSample = rangeBinsMoved
chirpSamp_givenRangeBin = signal_rfft[np.arange(numRamps)[None,:],:,rangeBinsToSample]

if (flagRBM == 1):
    """ Correcting for the Pi phase jump caused due to the Range bin Migration"""
    binDelta = np.abs(rangeBinsToSample[:,1::] - rangeBinsToSample[:,0:-1])
    tempVar = binDelta*np.pi
    binMigrationPhaseCorrTerm = np.zeros((rangeBinsToSample.shape),dtype=np.float32)
    binMigrationPhaseCorrTerm[:,1::] = tempVar
    binMigrationPhasorCorrTerm = np.exp(-1j*np.cumsum(binMigrationPhaseCorrTerm,axis=1))
    chirpSamp_givenRangeBin = chirpSamp_givenRangeBin*binMigrationPhasorCorrTerm[:,:,None]

    """ Correcting for the Doppler modulation caused due to the Range bin Migration"""
    rbmModulationAnalogFreq = (chirpBW/lightSpeed)*objectVelocity_mps
    dopplerBinOffset_rbm = (rbmModulationAnalogFreq/rampSamplingRate)*numDoppFFT
    # rbmModulationDigitalFreq = (rbmModulationAnalogFreq/rampSamplingRate)*2*np.pi
    # rbmModulationCorrectionTerm = np.exp(-1j*rbmModulationDigitalFreq[:,None]*np.arange(numRamps)[None,:])
    # chirpSamp_givenRangeBin = chirpSamp_givenRangeBin*rbmModulationCorrectionTerm[:,:,None]
else:
    dopplerBinOffset_rbm = np.zeros((numDopUniqRbin,))


objectVelocityBinNewScale = (objectVelocityBin/numRamps)*numDoppFFT
binOffset_Txphase = (phaseStepPerRamp_rad/(2*np.pi))*numDoppFFT
dopplerBinsToSample = np.round(objectVelocityBinNewScale[:,None] + dopplerBinOffset_rbm[:,None] + binOffset_Txphase[None,:]).astype('int32')
dopplerBinsToSample = np.mod(dopplerBinsToSample, numDoppFFT)


""" To obtain the signal coefficients, we could either do an FFT and evaluate at the required bins
or we could simply perform a single point DFT by correlating the signal with the phasor constructed out of the required bins.
In this commit, I'm evaluating the coefficients by doing a single point DFT at the known Doppler bins.
This way, we don't have to perform a huge FFT and then evaluate at the dopplerBinsToSample which is compute heavy.
Rather, we evaluate a single point DFT at the dopplerBinsToSample. T
This has a much smaller compute as compared to performing a huge FFT for 16 Rx channels.
Here, I'm computing the Doppler FFT for plotting purpose only"""
signalWindowed = chirpSamp_givenRangeBin*np.hanning(numRamps)[None,:,None]
signalFFT = np.fft.fft(signalWindowed, axis=1, n = numDoppFFT)/numRamps
signalFFTShift = signalFFT #np.fft.fftshift(signalFFT, axes= (0,))

""" Coefficients estimated using FFT and sampling the required bins"""
# mimoCoefficients_eachDoppler_givenRange = signalFFT[np.arange(numDopUniqRbin)[:,None],dopplerBinsToSample,:] # [numObj, numTx, numRx]

"""DFT based coefficient estimation """
DFT_vec = np.exp(1j*2*np.pi*(dopplerBinsToSample[:,None,:]/numDoppFFT)*np.arange(numRamps)[None,:,None])
mimoCoefficients_eachDoppler_givenRange = np.sum(signalWindowed[:,:,:,None]*np.conj(DFT_vec[:,:,None,:]),axis=1)/numRamps
mimoCoefficients_eachDoppler_givenRange = np.transpose(mimoCoefficients_eachDoppler_givenRange,(0,2,1))

mimoCoefficients_flatten = mimoCoefficients_eachDoppler_givenRange.reshape(-1, numTx_simult*numRx)
mimoCoefficients_flatten = mimoCoefficients_flatten[:,ulaInd]
ULA = np.unwrap(np.angle(mimoCoefficients_flatten),axis=1)
mimoCoefficients_flatten = mimoCoefficients_flatten*np.hanning(numMIMO)[None,:]
ULA_spectrum = np.fft.fft(mimoCoefficients_flatten,axis=1,n=numAngleFFT)/(numMIMO)
ULA_spectrum = np.fft.fftshift(ULA_spectrum,axes=(1,))
ULA_spectrumdB = 20*np.log10(np.abs(ULA_spectrum))
ULA_spectrumdB -= np.amax(ULA_spectrumdB,axis=1)[:,None]

signalFFTShiftSpectrum = np.abs(signalFFTShift)**2
# signalFFTShiftSpectrum = signalFFTShiftSpectrum/np.amax(signalFFTShiftSpectrum, axis=1)[:,None,:] # Normalize the spectrum for each Rx
signalMagSpectrum = 10*np.log10(np.abs(signalFFTShiftSpectrum))
powerMeanSpectrum_arossRxs = np.mean(signalFFTShiftSpectrum,axis=2) # Take mean spectrum across Rxs
noiseFloorEstFromSignal = 10*np.log10(np.percentile(powerMeanSpectrum_arossRxs,70,axis=1))
signalPowerDoppSpectrum = 10*np.log10(np.amax(powerMeanSpectrum_arossRxs,axis=1))
snrDoppSpectrum = signalPowerDoppSpectrum - noiseFloorEstFromSignal

DNL_rad = (DNL/180) * np.pi
noiseFloorSetByDNL = 10*np.log10((DNL_rad)**2/12) - 10*np.log10(numRamps) + 10*np.log10(numTx_simult) # DNL Noise floor raises as 10log10(numSimulTx)

print('\nSNR post Doppler FFT: {} dB'.format(np.round(snrDoppSpectrum)))
print('Noise Floor Estimated from signal: {} dB'.format(np.round(noiseFloorEstFromSignal)))
print('Noise Floor set by DNL: {} dB'.format(np.round(noiseFloorSetByDNL)))

plt.figure(1, figsize=(20,10),dpi=200)
plt.title('Power mean Range spectrum - Single chirp SNR = ' + str(np.round(binSNR)) + 'dB')
plt.plot(10*np.log10(signal_rfft_powermean) + dBFs_to_dBm)
# plt.axvline(rangeBinsToSample, color = 'k', linestyle = 'solid')
plt.xlabel('Range Bins')
plt.ylabel('Power dBm')
plt.grid(True)
plt.ylim([noiseFloor_perBin-10,0])

plt.figure(2, figsize=(20,10))
plt.suptitle('Doppler Spectrum with ' + str(numTx_simult) + 'Txs simultaneously ON in CDM')
for ele in range(numDopUniqRbin):
    plt.subplot(np.floor_divide(numDopUniqRbin-1,3)+1,min(3,numDopUniqRbin),ele+1)
    plt.plot(signalMagSpectrum[ele,:,0].T, lw=2, label='Target speed = ' + str(np.round(objectVelocity_mps[ele],2)) + ' mps') # Plotting only the 0th Rx instead of all 8
    # plt.vlines(dopplerBinsToSample[ele,:] ,ymin = -70, ymax = 10)
    plt.vlines(dopplerBinsToSample[ele,:],ymin = np.amin(noiseFloorEstFromSignal)-20, ymax = np.amax(signalPowerDoppSpectrum)+5)
    plt.xlabel('Doppler Bins')
    plt.ylabel('Power dBFs')
    plt.grid(True)
    plt.legend()


plt.figure(3, figsize=(20,10))
plt.suptitle('MIMO phase')
for ele in range(numDopUniqRbin):
    plt.subplot(np.floor_divide(numDopUniqRbin-1,3)+1,min(3,numDopUniqRbin),ele+1)
    plt.plot(ULA[ele,:], '-o')
    plt.xlabel('Rx #')
    plt.ylabel('Phase (rad)')
    plt.grid(True)


# plt.figure(4, figsize=(20,10), dpi=200)
plt.figure(4, figsize=(20,10))
plt.suptitle('MIMO ULA Angle spectrum')
for ele in range(numDopUniqRbin):
    plt.subplot(np.floor_divide(numDopUniqRbin-1,3)+1,min(3,numDopUniqRbin),ele+1)
    plt.plot(angAxis_deg, ULA_spectrumdB[ele,:],lw=2)
    plt.vlines(objectAzAngle_deg[ele], ymin = -70, ymax = 10)
    plt.xlabel('Angle (deg)')
    plt.ylabel('dB')
    plt.grid(True)
