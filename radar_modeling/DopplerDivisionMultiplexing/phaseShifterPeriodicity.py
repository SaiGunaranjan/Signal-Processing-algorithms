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
exactly. In such case, the same phase shifter code ia applied periodically
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
We also observe that the noise floor increases by 6 dB for every doubling of Tx
to the simultaneous transmission i.e., when we move from 2 Tx to 4 Tx, the noise floor raises by another 6 dB.
This is not very clear to me and I need to understand this better!!"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

numBitsPhaseShifter = 5
numPhaseCodes = 2**numBitsPhaseShifter
DNL = 360/(numPhaseCodes) # DNL in degrees
numTx_simult = 4
""" With 30 deg we see periodicity since 30 divides 360 but with say 29 deg, it doesn't divide 360 and hence periodicity is significantly reduced"""
phaseStepPerTx_deg = 29 # 29
phaseStepPerRamp_deg = np.arange(numTx_simult)*phaseStepPerTx_deg # Phase step per ramp per Tx
phaseStepPerRamp_rad = (phaseStepPerRamp_deg/360)*2*np.pi

numRamps = 140
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
to scale the alpha for other step sizes. Fow now, I have removed the quadratic term by setting alpha to 0"""


alpha = 0 # 1e-4
rampPhaseIdeal_deg = phaseStepPerRamp_deg[:,None]*(np.arange(numRamps)[None,:] + (alpha/2)*(np.arange(numRamps)[None,:])**2)

# alphaPerTx = (phaseStepPerRamp_deg/30) * 1e-4
# rampPhaseIdeal_deg = phaseStepPerRamp_deg[:,None]*(np.arange(numRamps)[None,:] + (alphaPerTx[:,None]/2)*(np.arange(numRamps)[None,:])**2)

rampPhaseIdeal_degWrapped = np.mod(rampPhaseIdeal_deg, 360)

phaseCodesIndexToBeApplied = np.argmin(np.abs(rampPhaseIdeal_degWrapped[:,:,None] - phaseShifterCodes_withNoise[None,None,:]),axis=2)
phaseCodesToBeApplied = phaseShifterCodes_withNoise[phaseCodesIndexToBeApplied]

phaseCodesToBeApplied_rad = (phaseCodesToBeApplied/180) * np.pi


signal = np.sum(np.exp(1j*phaseCodesToBeApplied_rad), axis=0)


signalWindowed = signal*np.hanning(numRamps)
signalFFT = np.fft.fft(signalWindowed)/numRamps
signalFFTShift = signalFFT #np.fft.fftshift(signalFFT)
signalFFTShiftSpectrum = np.abs(signalFFTShift)**2
signalFFTShiftSpectrum = signalFFTShiftSpectrum/np.amax(signalFFTShiftSpectrum)
signalMagSpectrum = 10*np.log10(np.abs(signalFFTShiftSpectrum))

noiseFloorSetByDNL = 10*np.log10((DNL/180 *np.pi)**2/12) - 10*np.log10(numRamps)
# noiseFloorEstFromSignal = 10*np.log10(np.mean(np.sort(signalFFTShiftSpectrum)[0:numRamps-10*numTx_simult]))
noiseFloorEstFromSignal = 10*np.log10(np.percentile(np.sort(signalFFTShiftSpectrum),65))

print('Noise Floor Estimated from signal with {} Txs simulataneously ON with phase sweeps: {} dB'.format(numTx_simult, \
                                                                                                         np.round(noiseFloorEstFromSignal)))
print('Noise Floor set by DNL: {} dB'.format(np.round(noiseFloorSetByDNL)))

binOffset_Txphase = (phaseStepPerRamp_rad/(2*np.pi))*numRamps
dopplerBinsToSample = np.round(binOffset_Txphase).astype('int32')
dopplerBinsToSample = np.mod(dopplerBinsToSample, numRamps)

plt.figure(1, figsize=(20,10),dpi=200)
# plt.title('Doppler Spectrum: Floor set by DNL = ' + str(np.round(noiseFloorSetByDNL)) + ' dB/bin')
plt.title('Doppler Spectrum with ' + str(numTx_simult) + 'Txs simultaneously ON in CDM')
plt.plot(signalMagSpectrum, lw=2)
# plt.vlines(dopplerBinsToSample,ymin = -70, ymax = 10)
plt.axhline(noiseFloorEstFromSignal, color = 'k', linestyle = 'solid')
plt.axhline(noiseFloorSetByDNL, color = 'k', linestyle = '-.')
plt.legend(['Doppler Spectrum', 'Noise floor Est. from spectrum', 'Theoretical Noise floor set by DNL'])
plt.xlabel('Doppler Bins')
plt.ylabel('Power dBFs')
plt.grid(True)



