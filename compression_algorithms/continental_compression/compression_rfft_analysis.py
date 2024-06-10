# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:36:29 2021

@author: saiguna
"""

"""
This script compares the SNR without any compression and with Conti compression schem.
"""



import numpy as np
import matplotlib.pyplot as plt
from conti_compression import ContiCompression



plt.close('all')

# np.random.seed(3)


""" Chirp Parameters"""
numTx_simult = 4
numRx = 4 # Increased number of Rxs from 4 to 32 to get a smoother averaging of signal/noise power
numRamps = 512
numSamp = 2048 # Number of ADC time domain samples
numSampPostRfft = 3 # Processing only 3 range bins. Since others are of no use. Saves compute.
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

""" Compression engine init"""
numBitsRangeFFTOutput = 32
numFracBitsRangeFFTOutput = numBitsRangeFFTOutput-1
numMantissaBits = 7 #4
contCompr = ContiCompression(numBitsRangeFFTOutput,numMantissaBits,numRx)


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
objectRange = 1*rangeRes # Fixing the taregt range bin to range bin = 1 out of the total 3 range bins for processing
objectVelocity_mps = 25*velocityRes
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

rampPhaseIdeal_deg = phaseStepPerRamp_deg[:,None]*(np.arange(numRamps)[None,:])
rampPhaseIdeal_degWrapped = np.mod(rampPhaseIdeal_deg, 360)
phaseCodesIndexToBeApplied = np.argmin(np.abs(rampPhaseIdeal_degWrapped[:,:,None] - phaseShifterCodes_withNoise[None,None,:]),axis=2)
phaseCodesToBeApplied = phaseShifterCodes_withNoise[phaseCodesIndexToBeApplied]
phaseCodesToBeApplied_rad = (phaseCodesToBeApplied/180) * np.pi

objectVelocityBinNewScale = (objectVelocityBin/numRamps)*numDoppFFT
binOffset_Txphase = (phaseStepPerRamp_rad/(2*np.pi))*numDoppFFT
dopplerBinsToSample = np.round(objectVelocityBinNewScale + binOffset_Txphase).astype('int32')
dopplerBinsToSample = np.mod(dopplerBinsToSample, numDoppFFT)

""" Correcting for the Pi phase jump caused due to the Range bin Migration"""
rangeBinsToSample = rangeBinsMoved
binDelta = np.abs(rangeBinsToSample[1::] - rangeBinsToSample[0:-1])
tempVar = binDelta*np.pi
binMigrationPhaseCorrTerm = np.zeros((rangeBinsToSample.shape),dtype=np.float32)
binMigrationPhaseCorrTerm[1::] = tempVar
binMigrationPhasorCorrTerm = np.exp(-1j*np.cumsum(binMigrationPhaseCorrTerm))

""" Correcting for the Doppler modulation caused due to the Range bin Migration"""
rbmModulationAnalogFreq = (chirpBW/lightSpeed)*objectVelocity_mps
rbmModulationDigitalFreq = (rbmModulationAnalogFreq/rampSamplingRate)*2*np.pi
rbmModulationCorrectionTerm = np.exp(-1j*rbmModulationDigitalFreq*np.arange(numRamps))

rfftBinSNRArray = np.arange(-15,70,4)#np.array([7, 69])#np.arange(-15,70,4)
numCases = len(rfftBinSNRArray)
doppNoiseFloorArr = np.zeros((numCases,),dtype=np.float32)
doppSignalPowerArr = np.zeros((numCases,),dtype=np.float32)
noiseFloorSetByDNLArr = np.zeros((numCases,),dtype=np.float32)
doppSpec = np.zeros((numCases,numDoppFFT),dtype=np.float32)

doppSpecdecomp = np.zeros((numCases,numDoppFFT),dtype=np.float32)
doppSignalPowerArrdecomp = np.zeros((numCases,),dtype=np.float32)
doppNoiseFloorArrdecomp = np.zeros((numCases,),dtype=np.float32)

signExtensionBits = np.zeros((numSampPostRfft,numRamps,2*numRx,numCases),dtype=np.uint32)

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

    signal_rangeWin = signal
    signal_rfft = np.fft.fft(signal_rangeWin,axis=2)/numSamp
    signal_rfft = signal_rfft[:,:,0:numSampPostRfft]
    """ Quantize RFFT"""
    signal_rfftQuant = np.floor(signal_rfft.real * 2**numFracBitsRangeFFTOutput + 0*0.5) + \
        1j*np.floor(signal_rfft.imag * 2**numFracBitsRangeFFTOutput + 0*0.5)
    signal_rfft = signal_rfftQuant/(2**numFracBitsRangeFFTOutput)

    # signal_rfft_powermean = np.mean(np.abs(signal_rfft)**2,axis=(0,1))

    chirpSamp_givenRangeBin = signal_rfft[np.arange(numRamps),:,rangeBinsToSample]
    chirpSamp_givenRangeBin = chirpSamp_givenRangeBin*binMigrationPhasorCorrTerm[:,None]
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



    """ With Conti Compression/decompression"""
    signal_rfftQuantReal = np.real(signal_rfftQuant).astype(np.uint32) # required in 2's complement format and hence typecast as uint
    signal_rfftQuantImag = np.imag(signal_rfftQuant).astype(np.uint32)

    signal_rfftQuantRealdecomp = np.zeros((numRamps,numRx,numSampPostRfft),dtype=np.uint32)
    signal_rfftQuantImagdecomp = np.zeros((numRamps,numRx,numSampPostRfft),dtype=np.uint32)
    for ele1 in range(numSampPostRfft):
        for ele2 in range(numRamps):
            rxSamplesReal, rxSamplesImag = signal_rfftQuantReal[ele2,:,ele1], signal_rfftQuantImag[ele2,:,ele1]
            compressedData = contCompr.compress_rx_samples(rxSamplesReal,rxSamplesImag)
            signExtensionBits[ele1,ele2,:,ele] = contCompr.BlockShiftArray
            contCompr.decompress_rx_samples(compressedData)
            signal_rfftQuantRealdecomp[ele2,:,ele1], signal_rfftQuantImagdecomp[ele2,:,ele1] = contCompr.rxSamplesRealRecon, contCompr.rxSamplesImagRecon

    signal_rfftQuantRealdecomp = signal_rfftQuantRealdecomp.astype(np.int32)
    signal_rfftQuantImagdecomp = signal_rfftQuantImagdecomp.astype(np.int32)

    signal_rfft_decomp = signal_rfftQuantRealdecomp + 1j*signal_rfftQuantImagdecomp
    signal_rfft_decomp = signal_rfft_decomp/(2**numFracBitsRangeFFTOutput)

    # signal_rfft_decomp_powermean = np.mean(np.abs(signal_rfft_decomp)**2,axis=(0,1))

    chirpSamp_givenRangeBin_decomp = signal_rfft_decomp[np.arange(numRamps),:,rangeBinsToSample]
    chirpSamp_givenRangeBin_decomp = chirpSamp_givenRangeBin_decomp*binMigrationPhasorCorrTerm[:,None]
    chirpSamp_givenRangeBin_decomp = chirpSamp_givenRangeBin_decomp*rbmModulationCorrectionTerm[:,None]

    signalWindoweddecomp = chirpSamp_givenRangeBin_decomp#*np.hanning(numRamps)[:,None]
    signalFFTdecomp = np.fft.fft(signalWindoweddecomp, axis=0, n = numDoppFFT)/numRamps
    signalFFTShiftdecomp = signalFFTdecomp
    signalFFTShiftSpectrumdecomp = np.abs(signalFFTShiftdecomp)**2
    signalMagSpectrumdecomp = 10*np.log10(np.abs(signalFFTShiftSpectrumdecomp))

    powerMeanSpectrum_arossRxsdecomp = np.mean(signalFFTShiftSpectrumdecomp,axis=1) # Take mean spectrum across Rxs
    noiseFloorEstFromSignaldecomp = 10*np.log10(np.mean(powerMeanSpectrum_arossRxsdecomp[200:450]))#10*np.log10(np.percentile(powerMeanSpectrum_arossRxs,70))
    signalPowerDoppSpectrumdecomp = 10*np.log10(np.amax(powerMeanSpectrum_arossRxsdecomp[dopplerBinsToSample]))
    snrDoppSpectrumdecomp = signalPowerDoppSpectrumdecomp - noiseFloorEstFromSignaldecomp

    doppSpecdecomp[ele,:] = 10*np.log10(powerMeanSpectrum_arossRxsdecomp)
    doppSignalPowerArrdecomp[ele] = signalPowerDoppSpectrumdecomp
    doppNoiseFloorArrdecomp[ele] = noiseFloorEstFromSignaldecomp


    print('{0} / {1} cases completed'.format(ele+1, numCases))

measuredSNRPostDoppFFT = doppSignalPowerArr - doppNoiseFloorArr
trueSNRPostDoppFFT = rfftBinSNRArray + 10*np.log10(numRamps)
measuredSNRPostDoppFFTdecomp = doppSignalPowerArrdecomp - doppNoiseFloorArrdecomp

vals, counts = np.unique(signExtensionBits.flatten(),return_counts=True)

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
plt.plot(doppSpec[2,:], lw=2,label='without compression')
plt.plot(doppSpecdecomp[2,:], lw=2,alpha=0.5,label='with Conti compression/decompression')
plt.axhline(thermalNoiseFloorPostDFFT, color = 'k', linestyle = 'solid',label = 'Thermal Noise floor')
plt.axhline(noiseFloorSetByDNLArr[2], color = 'k', linestyle = '-.',label='Phase quant. dBc noise floor')
plt.legend()
plt.xlabel('Doppler Bins')
plt.ylabel('Power dBFs')
plt.grid(True)
plt.ylim([thermalNoiseFloorPostDFFT+dBcnoiseFloorSetByDNL, 0])
plt.subplot(1,2,2)
plt.title('RFFT SNR = {} dB'.format(rfftBinSNRArray[-1]))
plt.plot(doppSpec[-1,:], lw=2,label='without compression')
plt.plot(doppSpecdecomp[-1,:], lw=2,alpha=0.5,label='with Conti compression/decompression')
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
plt.plot(rfftBinSNRArray,doppNoiseFloorArr, '-o',label='without compression')
plt.plot(rfftBinSNRArray,doppNoiseFloorArrdecomp, '-o',alpha=0.5,label='with Conti compression/decompression')
plt.axhline(thermalNoiseFloorPostDFFT, color = 'k', linestyle = 'solid',label = 'Expected Doppler thermal Noise floor')
plt.xlabel('RFFT SNR (dB)')
plt.ylabel('Dopp Floor (dB)')
plt.legend()
plt.grid(True)
plt.subplot(1,2,2)
plt.title('Measured SNR vs Expected SNR (post Dopp FFT)')
plt.plot(trueSNRPostDoppFFT,measuredSNRPostDoppFFT, '-o',label='without compression')
plt.plot(trueSNRPostDoppFFT,measuredSNRPostDoppFFTdecomp, '-o',alpha=0.5,label='with Conti compression/decompression')
plt.axhline(np.abs(dBcnoiseFloorSetByDNL),color='k', linestyle='dashed',label='PhaseShifter limited dynamic range = {0:.0f} dB'.format(abs(dBcnoiseFloorSetByDNL)))
plt.axvline(np.abs(dBcnoiseFloorSetByDNL),color='k', linestyle='dashed')
plt.xlabel('Expected SNR dB (post Dopp FFT)')
plt.ylabel('Measured SNR dB (post Dopp FFT)')
plt.legend()
plt.grid(True)


# plt.figure(4,figsize=(20,10),dpi=200)
# plt.title('Histogram of blockshifts across all channels, ramps, range bins, SNR')
# plt.plot(vals,counts,'-o')
# plt.xlabel('Block shifts/sign extension bits')
# plt.ylabel('Counts')
# plt.grid(True)


# plt.figure(5,figsize=(20,10))
# plt.title('Block shifts vs ramps for target range bin')
# plt.plot(np.amin(signExtensionBits[30,:,:,0],axis=1),'-o',label='low SNR target')
# plt.plot(np.amin(signExtensionBits[30,:,:,-1],axis=1),'-o',label='high SNR target')
# plt.xlabel('Ramps')
# plt.ylabel('Block shifts')
# plt.legend()
# plt.grid(True)

