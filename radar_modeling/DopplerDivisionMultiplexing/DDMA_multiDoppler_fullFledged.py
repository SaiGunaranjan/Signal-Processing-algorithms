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

""" This script now supports 3 methods of MIMO coefficient estimation for the DDMA scheme:
    1. FFT of the chirp samples + Tx phase modulated Doppler bin sampling
    2. DFT of the chirp samples with the Tx phase modulated Doppler frequencies
    3. Demodulation of the DDMA chirp sample data with the phase code sequence of each TX
    followed by DFT with the baseband Doppler frequencies
"""

"""
Introduced Tx-Tx coupling into the DDMA model

In this script, I have introduced inter-Tx coupling model into the DDMA scheme. When we have nearby Txs
simutaneously transmitting different signals each, there could be coupling from adjacent (and other nearby) Txs
thus corrupting the original signal transmitted by a particular Tx. This coupling has both a magnitude component
and a phase component. Let us consider a simple signgle IC with 4 Txs say, Tx0, Tx1, Tx2, Tx3.
Based on measurements results, typically, the coupling from adjacent Txs i.e Tx1 onto Tx0
(and similarly from Tx2 onto Tx1, Tx3 onto Tx2) is about 20 dB and from Tx2 to Tx0 (similarly from Tx3 to Tx1) is
a further 6 dB lower. In other words, when you have all 4 Txs, i.e. Tx0, Tx1, Tx2, Tx3 all ON simultaneously and
transmitting different signals each, in addition to the signal transmitted by Tx0, the signal from Tx1 couples onto Tx0
and is 20 dB lower in power. Similarly, the signal from Tx2 couples onto Tx0(in power) and is 20 + 6 dB lower,
signal from Tx3 couples onto Tx0 and is 20 + 6 + 6 dB lower and so on. Roughly drops by a further 6 dB (or even lower)
there onwards.  Hence the coupling from Tx1 to Tx0 on the linear voltage scale is 0.1,
from Tx2 to Tx0 is 0.05 (on linear voltage scale), from Tx3 to Tx0 is 0.025 and so on.
This is the magitude/amplitude coupling. On top of this, we could also have a random phase contribution from neighbouring Txs.
This can be captured as a caliberation in the DDMA mode and can be applied at the receiver end to remove this effect.
I have observed that the inter Tx coupling affects the SLLs in the angle spectrum. To be more particular,
with the dB coupling numbers mentioned (20, 26, 32, ..), it is actually the un-compensated coupled random phase that
plays a bigger role in setting the SLLs than the magnitude coupling. So these random phases need to be calibeated out.

This is the Tx coupling model I have introduced. This plays a very important role in DDMA schemes. There are other factors which play a crucial role in the DDMA scheme like:
1. Non-linearity of the phase response of the phase LUT
2. Non-linearity of the magnitude response of the phase LUT
3. Bin shift
4. Phase shifter cascaded coupling.
5. Effective antenna pattern in DDMA scheme with multiple Txs simultaneously ON and sweeping phase
and so on. I will try to add these models into the DDMA scheme one by one.

"""

""" The derivation for the DDMA scheme is available in the below location:
    https://saigunaranjan.atlassian.net/wiki/spaces/RM/pages/1966081/Code+Division+Multiple+Access+in+FMCW+RADAR"""

import numpy as np
import matplotlib.pyplot as plt
from mimoPhasorSynthesis import mimoPhasorSynth # loading the antenna corodinates for the steradian RADAR platforms
from ddma_class import DDMA_Radar


plt.close('all')

""" Flags to determine setttings to be used in DDMA scheme"""
flagRBM = 1 # 1 to enabled range bin migration term and 0 to disable
flagEnableICCoupling = 1 # 1 to enable , 0 to disable
flagEnableAntennaCoupling = 0 # 1 to enable, 0 to disable
flagEnableBoreSightCal = 1 # 1 to enable boresight cal, 0 to disable boresight cal
phaseDemodMethod = 1 #  1 for Tx demodulation method, 0 for modulated Doppler based sampling,
platform = 'SRIR16' # 'SRIR16', 'SRIR256', 'SRIR144'

""" Initialize DDMA object"""
ddma_radar = DDMA_Radar(flagEnableICCoupling, flagEnableAntennaCoupling, platform, flagRBM, \
                 phaseDemodMethod, flagEnableBoreSightCal)

binSNR = 10 # dB


""" Define Phase shifter settings"""
ddma_radar.define_phaseShifter_settings()
""" Define targets"""
ddma_radar.target_definitions()
""" Introduce coupling"""
ddma_radar.introduce_coupling()
""" Generate DDMA signal"""
ddma_radar.ddma_signal_generation(binSNR)

""" DDMA signal processing"""
ddma_radar.ddma_range_processing()
""" Correct for Range bin migration induced phase jump and frequency drift"""
ddma_radar.rbm_phase_freqbin_correction()
""" MIMO coefficient estimation """
ddma_radar.mimo_coefficient_estimation()
""" Bore-sight caliberation"""
ddma_radar.extract_boresight_cal()
""" Angle estimation"""
ddma_radar.angle_estimation()



""" Processing for plots"""
signalFFTShift = ddma_radar.signalFFT #np.fft.fftshift(signalFFT, axes= (0,))
signalFFTShiftSpectrum = np.abs(signalFFTShift)**2
signalMagSpectrum = 10*np.log10(np.abs(signalFFTShiftSpectrum))
powerMeanSpectrum_arossRxs = np.mean(signalFFTShiftSpectrum,axis=2) # Take mean spectrum across Rxs
noiseFloorEstFromSignal = 10*np.log10(np.percentile(powerMeanSpectrum_arossRxs,70,axis=1))
signalPowerDoppSpectrum = 10*np.log10(np.amax(powerMeanSpectrum_arossRxs,axis=1))
snrDoppSpectrum = signalPowerDoppSpectrum - noiseFloorEstFromSignal


print('\nSNR post Doppler FFT: {} dB'.format(np.round(snrDoppSpectrum)))
print('Noise Floor Estimated from signal: {} dB'.format(np.round(noiseFloorEstFromSignal)))
print('Noise Floor set by DNL: {} dB'.format(np.round(ddma_radar.noiseFloorSetByDNL)))

plt.figure(1, figsize=(20,10),dpi=200)
plt.title('Power mean Range spectrum - Single chirp SNR = ' + str(np.round(binSNR)) + 'dB')
plt.plot(10*np.log10(ddma_radar.signal_rfft_powermean) + ddma_radar.dBFs_to_dBm)
# plt.axvline(rangeBinsToSample, color = 'k', linestyle = 'solid')
plt.xlabel('Range Bins')
plt.ylabel('Power dBm')
plt.grid(True)
plt.ylim([ddma_radar.noiseFloor_perBin-10,0])


plt.figure(2, figsize=(20,10))
if (phaseDemodMethod == 1):
    plt.suptitle('Doppler Spectrum with ' + str(ddma_radar.numTx_simult) + 'Txs simultaneously ON in CDM')
    subplotCount = 0
    for ele in range(ddma_radar.numDopUniqRbin):
        for ele_tx in np.arange(ddma_radar.numTx_simult):
            plt.subplot(min(3,ddma_radar.numDopUniqRbin),ddma_radar.numTx_simult, subplotCount+1)
            if (ele==0):
                plt.title('Tx' + str(ele_tx) + ' demodulated spectrum')
            plt.plot(signalMagSpectrum[ele,:,0,ele_tx].T, lw=2, label='Target speed = ' + str(np.round(ddma_radar.objectVelocity_mps[ele],2)) + ' mps') # Plotting only the 0th Rx instead of all 8
            plt.vlines(ddma_radar.dopplerBinsToSample[ele,0],ymin = np.amin(noiseFloorEstFromSignal)-20, ymax = np.amax(signalPowerDoppSpectrum)+5)
            if (ele == ddma_radar.numDopUniqRbin-1):
                plt.xlabel('Doppler Bins')
            if (ele_tx==0):
                plt.ylabel('Power dBFs')
            plt.grid(True)
            plt.legend()
            subplotCount += 1

else:
    plt.suptitle('Doppler Spectrum with ' + str(ddma_radar.numTx_simult) + 'Txs simultaneously ON in CDM')
    for ele in range(ddma_radar.numDopUniqRbin):
        plt.subplot(np.floor_divide(ddma_radar.numDopUniqRbin-1,3)+1,min(3,ddma_radar.numDopUniqRbin),ele+1)
        plt.plot(signalMagSpectrum[ele,:,0].T, lw=2, label='Target speed = ' + str(np.round(ddma_radar.objectVelocity_mps[ele],2)) + ' mps') # Plotting only the 0th Rx instead of all 8
        plt.vlines(ddma_radar.dopplerBinsToSample[ele,:],ymin = np.amin(noiseFloorEstFromSignal)-20, ymax = np.amax(signalPowerDoppSpectrum)+5)
        plt.xlabel('Doppler Bins')
        plt.ylabel('Power dBFs')
        plt.grid(True)
        plt.legend()


plt.figure(3, figsize=(20,10))
plt.suptitle('MIMO phase')
for ele in range(ddma_radar.numDopUniqRbin):
    plt.subplot(np.floor_divide(ddma_radar.numDopUniqRbin-1,3)+1,min(3,ddma_radar.numDopUniqRbin),ele+1)
    plt.plot(ddma_radar.ULA[ele,:], '-o')
    plt.xlabel('Rx #')
    plt.ylabel('Phase (rad)')
    plt.grid(True)



plt.figure(4, figsize=(20,10))
plt.suptitle('MIMO ULA Angle spectrum')
for ele in range(ddma_radar.numDopUniqRbin):
    plt.subplot(np.floor_divide(ddma_radar.numDopUniqRbin-1,3)+1,min(3,ddma_radar.numDopUniqRbin),ele+1)
    plt.plot(ddma_radar.angAxis_deg, ddma_radar.ULA_spectrumdB[ele,:],lw=2)
    plt.vlines(ddma_radar.objectAzAngle_deg[ele], ymin = -70, ymax = 10)
    plt.xlabel('Angle (deg)')
    plt.ylabel('dB')
    plt.grid(True)
    plt.ylim([-70,10])
