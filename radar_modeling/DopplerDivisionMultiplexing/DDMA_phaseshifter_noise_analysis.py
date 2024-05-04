# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:36:29 2021

@author: saiguna
"""

"""
Multiplicative noise or phase noise appears as a product term to the signal where as additive noise appears
as an additive term to the signal.
Multiplicative noise --> x[n] = s[n] * w[n]
Additive noise       --> x[n] = s[n] + w[n]
Since the phase noise appears as a product to the signal term, as the signal amplitude increases,
the phase noise also starts scaling.

In this script, I have shown how the phase shifter quantization noise (PSQN) appears as a
multiplicative noise (phase noise) and hence limits the SNR for strong targets. More specifically,
I will show that the phase shifter quantization noise is a dBc noise (or phase noise or multiplicative noise)
and scales with the signal power beyond a point. For weaker SNR targets, the phase noise floor
will be buried below the thermal floor (post Doppler FFT). When the SNR of the targets increases
and reaches >=58 dB (PSQN floor for 512 ramps), the SNR will saturate to the phase noise i.e 58 dBc (in this case).

1. For low SNR targets, the PSQN is buried below the thermal floor(post Doppler FFT).
2. For very strong targets (Doppler FFT SNR >> 58 dBc), the PSQN limits the SNR. So, targets having
Doppler FFT SNR >= 58dBc, will all report the same SNR.
3. Mathematically, this can be written as:
SNR_PSQN  = SNR ; SNR < PSQN
            PSQN; SNR ≥ PSQN
	where,
	 SNR_PSQN is Doppler FFT SNR with phase shifter quantization noise(PSQN) for 4 Tx DDMA system,
	 SNR is the Doppler FFT SNR without PSQN
	 PSQN = 58 dBc for 512 ramps, 4 TX DDMA system with 7 bit phase shifter covering 360°

Phase shifter Quantization noise power(for 4 Tx) spread over 512 ramps/Doppler bins = (Δ^2 / 12) * (4 / 512)
PSQN (dBc) = 10log_10 ((Δ^2 / 12) * (4 / 512)) ~ 58 dBc
"""



import numpy as np
import matplotlib.pyplot as plt



plt.close('all')
np.random.seed(3) # 3

""" Chirp Parameters"""
numTx_simult = 4
numRx = 32 # Increased number of Rxs from 4 to 32 to get a smoother averaging of signal/noise power
numRamps = 512
numSamp = 2048 # Number of ADC time domain samples
numSampPostRfft = numSamp//2
lightSpeed = 3e8
numDoppFFT = 512#2048
chirpBW = 1e9 # Hz
centerFreq = 76.5e9 # GHz
interRampTime = 44e-6 # us
rampSamplingRate = 1/interRampTime
rangeRes = lightSpeed/(2*chirpBW)
maxRange = numSampPostRfft*rangeRes # m
lamda = lightSpeed/centerFreq

""" Derived Parameters """
chirpSamplingRate = 1/interRampTime
maxVelBaseband_mps = (chirpSamplingRate/2) * (lamda/2) # m/s
FsEquivalentVelocity = 2*maxVelBaseband_mps # Fs = 2*Fs/2
velocityRes = (chirpSamplingRate/numRamps) * (lamda/2)



## RF parameters
thermalNoise = -174 # dBm/Hz
noiseFigure = 9 # dB
baseBandgain = 38 #dB
adcSamplingRate = 56.25e6 # 56.25 MHz
adcSamplingTime = 1/adcSamplingRate # s
chirpOnTime = numSamp*adcSamplingTime
chirpSlope = chirpBW/chirpOnTime
dBFs_to_dBm = 10
totalNoisePower_dBm = thermalNoise + noiseFigure + baseBandgain + 10*np.log10(adcSamplingRate)
totalNoisePower_dBFs = totalNoisePower_dBm - 10
noiseFloor_perBin = totalNoisePower_dBFs - 10*np.log10(numSamp) # dBFs/bin
noisePower_perBin = 10**(noiseFloor_perBin/10)
totalNoisePower = noisePower_perBin*numSamp # sigmasquare totalNoisePower
sigma = np.sqrt(totalNoisePower)

thermalNoiseFloorPostDFFT = noiseFloor_perBin - 10*np.log10(numRamps)




""" Target definition"""
objectRange = np.random.uniform(10,maxRange-10) # 60.3 # m
objectVelocity_mps = 0
objectAzAngle_deg = np.random.uniform(-50,50)
objectAzAngle_rad = (objectAzAngle_deg/360) * (2*np.pi)

objectVelocity_baseBand_mps = np.mod(objectVelocity_mps, FsEquivalentVelocity) # modulo Fs [from 0 to Fs]
objectVelocityBin = objectVelocity_baseBand_mps/velocityRes
objectRangeBin = objectRange/rangeRes
rangeMoved = objectRange + objectVelocity_mps*interRampTime*np.arange(numRamps)
rangeBinsMoved = np.floor(rangeMoved/rangeRes).astype('int32')



""" Phase shifter settings"""
""" With 30 deg, we see periodicity since 30 divides 360 but with say 29 deg, it doesn't divide 360 and hence periodicity is significantly reduced"""
numBitsPhaseShifter = 7
numPhaseCodes = 2**numBitsPhaseShifter
DNL = 360/(numPhaseCodes) # DNL in degrees
DNL_rad = (DNL/180) * np.pi
dBcnoiseFloorSetByDNL = 10*np.log10(((DNL_rad)**2)/12) - 10*np.log10(numRamps) + 10*np.log10(numTx_simult) # DNL Noise floor raises as 10log10(numSimulTx)

phaseStepPerTx_deg = 29#29.3
phaseStepPerRamp_deg = np.array([ 0.      , 28.828125, 57.65625 , 86.484375])#np.arange(numTx_simult)*phaseStepPerTx_deg # Phase step per ramp per Tx
phaseStepPerRamp_rad = (phaseStepPerRamp_deg/360)*2*np.pi

phaseShifterCodes = DNL*np.arange(numPhaseCodes)
phaseShifterNoise = np.random.uniform(-DNL/2, DNL/2, numPhaseCodes)
phaseShifterCodes_withNoise = phaseShifterCodes + phaseShifterNoise

""" Ensure that the phase shifter LUT is without any bias and is from 0 to 360 degrees"""
phaseShifterCodes_withNoise = np.mod(phaseShifterCodes_withNoise,360)
alpha = 0#1e-4
rampPhaseIdeal_deg = phaseStepPerRamp_deg[:,None]*(np.arange(numRamps)[None,:] + (alpha/2)*(np.arange(numRamps)[None,:])**2)
rampPhaseIdeal_degWrapped = np.mod(rampPhaseIdeal_deg, 360)
phaseCodesIndexToBeApplied = np.argmin(np.abs(rampPhaseIdeal_degWrapped[:,:,None] - phaseShifterCodes_withNoise[None,None,:]),axis=2)
phaseCodesToBeApplied = phaseShifterCodes_withNoise[phaseCodesIndexToBeApplied]
phaseCodesToBeApplied_rad = (phaseCodesToBeApplied/180) * np.pi

objectVelocityBinNewScale = (objectVelocityBin/numRamps)*numDoppFFT
binOffset_Txphase = (phaseStepPerRamp_rad/(2*np.pi))*numDoppFFT
dopplerBinsToSample = np.round(objectVelocityBinNewScale + binOffset_Txphase).astype('int32')
dopplerBinsToSample = np.mod(dopplerBinsToSample, numDoppFFT)

rfftBinSNRArray = np.arange(-15,70,4)
numCases = len(rfftBinSNRArray)
doppNoiseFloorArr = np.zeros((numCases,),dtype=np.float32)
doppSignalPowerArr = np.zeros((numCases,),dtype=np.float32)
noiseFloorSetByDNLArr = np.zeros((numCases,),dtype=np.float32)
doppSpec = np.zeros((numCases,numDoppFFT),dtype=np.float32)

for ele in range(numCases):

    binSNR = rfftBinSNRArray[ele]
    signalPowerdBFs = noiseFloor_perBin + binSNR
    signalPower = 10**(signalPowerdBFs/10)
    signalAmplitude = np.sqrt(signalPower)
    signalPhase = np.exp(1j*np.random.uniform(-np.pi, np.pi))
    signalphasor = signalAmplitude*signalPhase

    rangeTerm = signalphasor*np.exp(1j*((2*np.pi*objectRangeBin)/numSamp)*np.arange(numSamp))
    dopplerTerm = np.exp(1j*((2*np.pi*objectVelocityBin)/numRamps)*np.arange(numRamps))
    """ Range Bin migration term"""
    rangeBinMigration = \
        np.exp(1j*2*np.pi*chirpSlope*(2*objectVelocity_mps/lightSpeed)*interRampTime*adcSamplingTime*np.arange(numRamps)[:,None]*np.arange(numSamp)[None,:])


    rxSignal = np.exp(1j*2*np.pi*0.5*np.sin(objectAzAngle_rad)*np.arange(numRx))

    signal_phaseCode = np.exp(1j*phaseCodesToBeApplied_rad)
    phaseCodedTxSignal = dopplerTerm[None,:] * signal_phaseCode # [numTx, numRamps]
    phaseCodedTxRxSignal = phaseCodedTxSignal[:,:,None]*rxSignal[None,None,:] #[numTx, numRamps, numTx, numRx]
    phaseCodedTxRxSignal_withRangeTerm = rangeTerm[None,None,None,:] * phaseCodedTxRxSignal[:,:,:,None]
    """ Including the range bin migration term as well"""
    phaseCodedTxRxSignal_withRangeTerm = phaseCodedTxRxSignal_withRangeTerm * rangeBinMigration[None,:,None,:]
    signal = np.sum(phaseCodedTxRxSignal_withRangeTerm, axis=(0)) # [numRamps,numRx, numSamp]
    noise = (sigma/np.sqrt(2))*np.random.randn(numRamps*numRx*numSamp) + 1j*(sigma/np.sqrt(2))*np.random.randn(numRamps*numRx*numSamp)
    noise = noise.reshape(numRamps, numRx, numSamp)
    signal = signal + noise

    signal_rangeWin = signal#*np.hanning(numSamp)[None,None,:]
    signal_rfft = np.fft.fft(signal_rangeWin,axis=2)/numSamp
    signal_rfft = signal_rfft[:,:,0:numSampPostRfft]
    signal_rfft_powermean = np.mean(np.abs(signal_rfft)**2,axis=(0,1))


    rangeBinsToSample = rangeBinsMoved
    chirpSamp_givenRangeBin = signal_rfft[np.arange(numRamps),:,rangeBinsToSample]

    """ Correcting for the Pi phase jump caused due to the Range bin Migration"""
    binDelta = np.abs(rangeBinsToSample[1::] - rangeBinsToSample[0:-1])
    tempVar = binDelta*np.pi
    binMigrationPhaseCorrTerm = np.zeros((rangeBinsToSample.shape),dtype=np.float32)
    binMigrationPhaseCorrTerm[1::] = tempVar
    binMigrationPhasorCorrTerm = np.exp(-1j*np.cumsum(binMigrationPhaseCorrTerm))
    chirpSamp_givenRangeBin = chirpSamp_givenRangeBin*binMigrationPhasorCorrTerm[:,None]

    """ Correcting for the Doppler modulation caused due to the Range bin Migration"""
    rbmModulationAnalogFreq = (chirpBW/lightSpeed)*objectVelocity_mps
    rbmModulationDigitalFreq = (rbmModulationAnalogFreq/rampSamplingRate)*2*np.pi
    rbmModulationCorrectionTerm = np.exp(-1j*rbmModulationDigitalFreq*np.arange(numRamps))
    chirpSamp_givenRangeBin = chirpSamp_givenRangeBin*rbmModulationCorrectionTerm[:,None]

    signalWindowed = chirpSamp_givenRangeBin#*np.hanning(numRamps)[:,None]
    signalFFT = np.fft.fft(signalWindowed, axis=0, n = numDoppFFT)/numRamps
    signalFFTShift = signalFFT #np.fft.fftshift(signalFFT, axes= (0,))
    signalFFTShiftSpectrum = np.abs(signalFFTShift)**2
    signalMagSpectrum = 10*np.log10(np.abs(signalFFTShiftSpectrum))

    powerMeanSpectrum_arossRxs = np.mean(signalFFTShiftSpectrum,axis=1) # Take mean spectrum across Rxs
    noiseFloorEstFromSignal = 10*np.log10(np.mean(powerMeanSpectrum_arossRxs[200:450]))#10*np.log10(np.percentile(powerMeanSpectrum_arossRxs,70))
    signalPowerDoppSpectrum = 10*np.log10(np.amax(powerMeanSpectrum_arossRxs[dopplerBinsToSample]))
    snrDoppSpectrum = signalPowerDoppSpectrum - noiseFloorEstFromSignal
    noiseFloorSetByDNL = signalPowerDoppSpectrum + dBcnoiseFloorSetByDNL

    doppSpec[ele,:] = 10*np.log10(powerMeanSpectrum_arossRxs)
    doppSignalPowerArr[ele] = signalPowerDoppSpectrum
    doppNoiseFloorArr[ele] = noiseFloorEstFromSignal
    noiseFloorSetByDNLArr[ele] = noiseFloorSetByDNL


    # #print('\nSNR post Doppler FFT: {} dB'.format(np.round(snrDoppSpectrum)))
    # print('Actual Thermal Noise Floor post Doppler FFT: {} dB'.format(np.round(thermalNoiseFloorPostDFFT)))
    # print('Noise Floor Estimated from Doppler domain: {} dB'.format(np.round(noiseFloorEstFromSignal)))
    # print('Actual phase shifter noise floor: {} dB'.format(np.round(noiseFloorSetByDNL)))


    print('{0} / {1} cases completed'.format(ele+1, numCases))

print('\n\nNoise Floor set by {0} Tx phase shifter DNL: {1} dBc'.format(numTx_simult, np.round(dBcnoiseFloorSetByDNL)))

measuredSNRPostDoppFFT = doppSignalPowerArr - doppNoiseFloorArr
trueSNRPostDoppFFT = rfftBinSNRArray + 10*np.log10(numRamps)


# plt.figure(1, figsize=(20,10),dpi=200)
# plt.title('Range spectrum')
# plt.plot(10*np.log10(signal_rfft_powermean) + dBFs_to_dBm)
# plt.xlabel('Range Bins')
# plt.ylabel('Power dBm')
# plt.grid(True)


plt.figure(2, figsize=(20,10), dpi=150)
plt.suptitle('Effect of Phase noise on Doppler Spectrum with ' + str(numTx_simult) + 'Txs DDMA')
plt.subplot(1,2,1)
plt.title('RFFT SNR = {} dB'.format(rfftBinSNRArray[2]))
plt.plot(doppSpec[2,:], lw=2)
plt.axhline(thermalNoiseFloorPostDFFT, color = 'k', linestyle = 'solid',label = 'Thermal Noise floor')
plt.axhline(noiseFloorSetByDNLArr[2], color = 'k', linestyle = '-.',label='Phase quant. dBc noise floor')
plt.legend()
plt.xlabel('Doppler Bins')
plt.ylabel('Power dBFs')
plt.grid(True)
plt.ylim([thermalNoiseFloorPostDFFT+dBcnoiseFloorSetByDNL, 0])

plt.subplot(1,2,2)
plt.title('RFFT SNR = {} dB'.format(rfftBinSNRArray[-1]))
plt.plot(doppSpec[-1,:], lw=2)
plt.axhline(thermalNoiseFloorPostDFFT, color = 'k', linestyle = 'solid',label = 'Thermal Noise floor')
plt.axhline(noiseFloorSetByDNLArr[-1], color = 'k', linestyle = '-.',label='Phase quant. dBc noise floor')
plt.legend()
plt.xlabel('Doppler Bins')
plt.ylabel('Power dBFs')
plt.grid(True)
plt.ylim([thermalNoiseFloorPostDFFT+dBcnoiseFloorSetByDNL, 0])


plt.figure(3, figsize=(20,10),dpi=150)
plt.subplot(1,2,1)
plt.title('Dopp Floor vs RFFT SNR')
plt.plot(rfftBinSNRArray,doppNoiseFloorArr, '-o')
plt.axhline(thermalNoiseFloorPostDFFT, color = 'k', linestyle = 'solid',label = 'Expected Doppler thermal Noise floor')
plt.xlabel('RFFT SNR (dB)')
plt.ylabel('Dopp Floor (dB)')
plt.legend()
plt.grid(True)
plt.subplot(1,2,2)
plt.title('Measured SNR vs Expected SNR (post Dopp FFT)')
plt.plot(trueSNRPostDoppFFT,measuredSNRPostDoppFFT, '-o')
plt.axhline(np.abs(dBcnoiseFloorSetByDNL),color='k', linestyle='dashed',label='PhaseShifter limited dynamic range = {0:.0f} dB'.format(abs(dBcnoiseFloorSetByDNL)))
plt.axvline(np.abs(dBcnoiseFloorSetByDNL),color='k', linestyle='dashed')
plt.xlabel('Expected SNR dB (post Dopp FFT)')
plt.ylabel('Measured SNR dB (post Dopp FFT)')
plt.legend()
plt.grid(True)








