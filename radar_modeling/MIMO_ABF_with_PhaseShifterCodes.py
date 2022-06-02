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
each with its own phase code per ramp and have also been able to estimate MIMO coeficients"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

numTx_simult = 4
numRx = 8
numMIMO = numTx_simult*numRx
numDoppFFT = 2048
numAngleFFT = 2048
txSpacing = 2e-3
rxSpacing = 4*txSpacing
lightSpeed = 3e8
centerFreq = 76.5e9
lamda = lightSpeed/centerFreq
objectAzAngle_deg = 10
objectAzAngle_rad = (objectAzAngle_deg/360) * (2*np.pi)
rxSignal = np.exp(1j*(2*np.pi/lamda)*rxSpacing*np.sin(objectAzAngle_rad)*np.arange(numRx))

numBitsPhaseShifter = 5
numPhaseCodes = 2**numBitsPhaseShifter
DNL = 360/(numPhaseCodes) # DNL in degrees

""" With 30 deg, we see periodicity since 30 divides 360 but with say 29 deg, it doesn't divide 360 and hence periodicity is significantly reduced"""
phaseStepPerTx_deg = 29#29.3
phaseStepPerRamp_deg = np.arange(numTx_simult)*phaseStepPerTx_deg # Phase step per ramp per Tx
phaseStepPerRamp_rad = (phaseStepPerRamp_deg/360)*2*np.pi
numRamps = 280
phaseShifterCodes = DNL*np.arange(numPhaseCodes)
phaseShifterNoise = np.random.uniform(-DNL/2, DNL/2, numPhaseCodes)
phaseShifterCodes_withNoise = phaseShifterCodes + phaseShifterNoise

""" A small quadratic term is added to the linear phase term to break the periodicity.
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

# alphaPerTx = (phaseStepPerRamp_deg/30) * 1e-4
# rampPhaseIdeal_deg = phaseStepPerRamp_deg[:,None]*(np.arange(numRamps)[None,:] + (alphaPerTx[:,None]/2)*(np.arange(numRamps)[None,:])**2)

rampPhaseIdeal_degWrapped = np.mod(rampPhaseIdeal_deg, 360)

phaseCodesIndexToBeApplied = np.argmin(np.abs(rampPhaseIdeal_degWrapped[:,:,None] - phaseShifterCodes_withNoise[None,None,:]),axis=2)
phaseCodesToBeApplied = phaseShifterCodes_withNoise[phaseCodesIndexToBeApplied]

phaseCodesToBeApplied_rad = (phaseCodesToBeApplied/180) * np.pi

txSignal = np.exp(1j*(2*np.pi/lamda)*txSpacing*np.sin(objectAzAngle_rad)*np.arange(numTx_simult))
signal_phaseCode = np.exp(1j*phaseCodesToBeApplied_rad)
phaseCodedTxSignal = signal_phaseCode*txSignal[:,None] # [numTx, numRamps]
phaseCodedTxRxSignal = phaseCodedTxSignal[:,:,None]*rxSignal[None,None,:] #[numTx, numRamps, numTx, numRx]
signal = np.sum(phaseCodedTxRxSignal, axis=(0)) # [numRamps,numRx]



signalWindowed = signal*np.hanning(numRamps)[:,None]
signalFFT = np.fft.fft(signalWindowed, axis=0, n = numDoppFFT)/numRamps
signalFFTShift = signalFFT #np.fft.fftshift(signalFFT, axes= (0,))
signalFFTShiftSpectrum = np.abs(signalFFTShift)**2
signalFFTShiftSpectrum = signalFFTShiftSpectrum/np.amax(signalFFTShiftSpectrum, axis=0)[None,:] # Normalize the spectrum for each Rx
signalMagSpectrum = 10*np.log10(np.abs(signalFFTShiftSpectrum))

dopplerBinsToSample = np.round((phaseStepPerRamp_rad/(2*np.pi))*numDoppFFT).astype('int32')

DNL_rad = (DNL/180) * np.pi
noiseFloorSetByDNL = 10*np.log10((DNL_rad)**2/12) - 10*np.log10(numRamps)

powerMeanSpectrum_arossRxs = np.mean(signalFFTShiftSpectrum,axis=1) # Take mean spectrum across Rxs
noiseFloorEstFromSignal = 10*np.log10(np.percentile(np.sort(powerMeanSpectrum_arossRxs),70))

print('Noise Floor Estimated from signal: {} dB'.format(np.round(noiseFloorEstFromSignal)))
print('Noise Floor set by DNL: {} dB'.format(np.round(noiseFloorSetByDNL)))

plt.figure(1, figsize=(20,10), dpi=300)
# plt.title('Doppler Spectrum: Floor set by DNL = ' + str(np.round(noiseFloorSetByDNL)) + ' dB/bin')
plt.title('Doppler Spectrum with ' + str(numTx_simult) + 'Txs simultaneously ON in CDM')
plt.plot(signalMagSpectrum[:,0], lw=2) # Plotting only the 0th Rx instead of all 8
plt.vlines(dopplerBinsToSample,ymin = -70, ymax = 10)
plt.axhline(noiseFloorEstFromSignal, color = 'k', linestyle = 'solid')
plt.axhline(noiseFloorSetByDNL, color = 'k', linestyle = '-.')
plt.legend(['Doppler Spectrum', 'Noise floor Est. from spectrum', 'Theoretical Noise floor set by DNL'])
plt.xlabel('Bins')
plt.ylabel('Power dBFs')
plt.grid(True)


mimoCoefficients_eachDoppler_givenRange = signalFFT[dopplerBinsToSample,:] # numTx*numDopp x numRx
mimoCoefficients_flatten = (mimoCoefficients_eachDoppler_givenRange.T).reshape(-1,numTx_simult*numRx)
ULA = np.unwrap(np.angle(mimoCoefficients_flatten[0,:]))
digFreq = (ULA[-1] - ULA[0])/(numMIMO - 1)
est_ang = np.arcsin((digFreq/(2*np.pi))*lamda/txSpacing)*180/np.pi

Fs_spatial = lamda/txSpacing
angAxis_deg = np.arcsin(np.arange(-numAngleFFT//2, numAngleFFT//2)*(Fs_spatial/numAngleFFT))*180/np.pi

ULA_spectrum = np.fft.fft(mimoCoefficients_flatten[0,:],n=numAngleFFT)/(numMIMO)
ULA_spectrum = np.fft.fftshift(ULA_spectrum)


plt.figure(2, figsize=(20,10), dpi=300)
plt.subplot(1,2,1)
plt.title('MIMO phase')
plt.plot(ULA,'-o')
plt.xlabel('Rx')
plt.ylabel('Phase (rad)')
plt.grid(True)
plt.subplot(1,2,2)
plt.title('MIMO ULA spectrum')
plt.plot(angAxis_deg, 20*np.log10(np.abs(ULA_spectrum)),label='Angle spectrum')
plt.axvline(objectAzAngle_deg, color = 'k', label='Ground Truth angle (deg)')
plt.legend()
plt.xlabel('Angle')
plt.ylabel('dB')
plt.grid(True)


