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
"""

""" In this commit, I have also modeled the Txs/Rxs with simulataneous transmission from all Txs
each with its own phase code per ramp and have also been able to estimate MIMO coeficients.
I have modelled the DDMA for the Steradian SRIR144 and SRIR256 platforms.
"""

""" In addition to angle accuracy, the script now also checks SLLs in the angle spectrum"""

"""
Introduced Tx-Tx Rx-Rx IC level coupling into the DDMA model

In this script, I have introduced inter-Tx and inter-Rx coupling model into the DDMA scheme. When we have nearby Txs
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
plays a bigger role in setting the SLLs than the magnitude coupling. So these random phases need to be caliberated out.

Introduced antenna coupling in addition to the IC coupling

Previously, I had modelled the IC level coupling for the Txs and Rxs. 20 dB between adjacent Txs and Rxs and a 6 dB drop from there on.
In this commit, I have also added the coupling introduced by the antennas as well. So in essence there are 2 coupling factors:
1. IC level coupling
2. Antenna coupling
The typical antenna level coupling is about 15 dB if the separation is lambda/2 and
falls to about 24 dB when the separation is 2*lambda. So the Tx antennas separated by lambda/2 and adjacent to each other
have a coupling of 15 dB. The Rx antennas which are separated by 2*lambda and adjacent to each other have a coupling of 24 dB.
I have assumed a drop of 6 dB as we move away from the Tx antennas or the Rx antennas. I have assumed a random phase coupling
for the antenna coupling matrix as well. But instead of varying the phase each Monte Carlo run(like I do for the IC coupling),
for the antenna coupling, I initialize the magnitude and phase only once at the beginning. I dont vary this frame to frame.
In terms of modelling, the antenna coupling is also a matrix (for Txs antennas and Rx antennas) which is multipled
with the IC coupling matrix. So the transmitted signal is TxAntennaCouplingMatrix x TxICCouplingMatrix x Tx phase coded signal.
Similary, the received signal is RxAntennaCouplingMatrix x RxICCouplingMatrix x Rx signls. For the received signal,
I have introduced the coupling before adding noise. I might need to introduce the coupling after adding noise.

This is the coupling model I have introduced. This plays a very important role in DDMA schemes.
There are other factors which play a crucial role in the DDMA scheme like:
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
from ddma_class import DDMA_Radar
import time as time
from scipy.signal import argrelextrema


""" To fix the seed, enable in the ddma class"""

tstart = time.time()

plt.close('all')
""" Flags to determine setttings to be used in DDMA scheme"""
flagRBM = 1
flagEnableICCoupling = 1 # 1 to enable , 0 to disable
flagEnableAntennaCoupling = 0 # 1 to enable, 0 to disable
flagEnableBoreSightCal = 1 # 1 to enable boresight cal, 0 to disable boresight cal
phaseDemodMethod = 1 #  1 for Tx demodulation method, 0 for modulated Doppler based sampling,
platform = 'SRIR16' # 'SRIR16', 'SRIR256', 'SRIR144'

""" Initialize DDMA object"""
ddma_radar = DDMA_Radar(flagEnableICCoupling, flagEnableAntennaCoupling, platform, flagRBM, \
                 phaseDemodMethod, flagEnableBoreSightCal)


""" MonteCarlo Parameters"""
range_binSNRArray = np.arange(-20, 30, 4)#np.arange(-20, 30, 2)#np.arange(-20, 30, 4)#np.arange(-20, 30, 2)  # dB
numMonteCarloRuns = 100#100#100#50 # 1
numSnrMC = len(range_binSNRArray)
angleErrorMatrix_std = np.zeros((numSnrMC,))
angleErrorMatrix_percentile = np.zeros((numSnrMC,))

angleSLLMatrix_median = np.zeros((numSnrMC))
angleSLLMatrix_max = np.zeros((numSnrMC))
angleSLLMatrix_percentile = np.zeros((numSnrMC))

percentile = 80#75

count_snrMC = 0
for binSNR in range_binSNRArray:
    tstart_snr = time.time()
    errorAngArray = np.empty([0])
    angleSllArray = np.empty([0])
    for iter_num in np.arange(numMonteCarloRuns):

        """ Define Phase shifter settings"""
        ddma_radar.define_phaseShifter_settings() # This is also inside the montecarlo loop because, we randomize the phase shifter DNL for each run
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


        """ Angle statistics computation"""

        """ Angle error"""
        angInd = np.argmax(ddma_radar.ULA_spectrumdB,axis=1)
        estAngDeg = ddma_radar.angAxis_deg[angInd]
        errorAng = ddma_radar.objectAzAngle_deg - estAngDeg
        errorAngArray = np.hstack((errorAngArray,errorAng))

        """ SLL computation"""
        sllValdBc = np.zeros((ddma_radar.numDopUniqRbin),dtype=np.float32)
        for ele1 in np.arange(ddma_radar.numDopUniqRbin):
            localMaxInd = argrelextrema(ddma_radar.ULA_spectrumdB[ele1,:],np.greater,axis=0,order=2)[0] # Use this when spectrum is evaluated on a large oversampled scale
            # localMaxInd = argrelextrema(ddma_radar.ULA_spectrumdB[ele1,:],np.greater_equal,axis=0,order=1)[0] # Use this when spectrum is not evaluated on an oversampled scale.
            try:
                sllInd = np.argsort(ddma_radar.ULA_spectrumdB[ele1,localMaxInd])[-2] # 1st SLL
                sllValdBc[ele1] = ddma_radar.ULA_spectrumdB[ele1,localMaxInd[sllInd]]
            except IndexError:
                sllValdBc[ele1] = 0


        angleSllArray = np.hstack((angleSllArray,sllValdBc))

        # if any(np.abs(errorAng)>3):
        #     print('Im here')
        #     print('True Velocities (mps):', np.round(ddma_radar.objectVelocity_mps,2))
        #     print('Baseband Velocities (mps):', np.round(ddma_radar.objectVelocity_baseBand_mpsBipolar,2))
        #     print('True Angles (deg):', np.round(ddma_radar.objectAzAngle_deg,2))
        #     print('Estimated Angles (deg):', np.round(estAngDeg,2))

        #     plt.figure(4, figsize=(20,10))
        #     plt.suptitle('MIMO ULA Angle spectrum')
        #     for ele in range(ddma_radar.numDopUniqRbin):
        #         plt.subplot(np.floor_divide(ddma_radar.numDopUniqRbin-1,3)+1,min(3,ddma_radar.numDopUniqRbin),ele+1)
        #         plt.plot(ddma_radar.angAxis_deg, ddma_radar.ULA_spectrumdB[ele,:],lw=2)
        #         plt.vlines(ddma_radar.objectAzAngle_deg[ele], ymin = -170, ymax = -110)
        #         plt.xlabel('Angle (deg)')
        #         plt.ylabel('dB')
        #         plt.grid(True)

        # if any(np.abs(sllValdBc)<7):
        #     print('Im here')
        #     print('True Velocities (mps):', np.round(ddma_radar.objectVelocity_mps,2))
        #     print('Baseband Velocities (mps):', np.round(ddma_radar.objectVelocity_baseBand_mpsBipolar,2))
        #     print('True Angles (deg):', np.round(ddma_radar.objectAzAngle_deg,2))
        #     print('Estimated Angles (deg):', np.round(estAngDeg,2))
        #     print('Estimated SLLs (dBc):', np.round(sllValdBc,2))

            # plt.figure(4, figsize=(20,10))
            # plt.suptitle('MIMO ULA Angle spectrum')
            # for ele in range(ddma_radar.numDopUniqRbin):
            #     plt.subplot(np.floor_divide(ddma_radar.numDopUniqRbin-1,3)+1,min(3,ddma_radar.numDopUniqRbin),ele+1)
            #     plt.plot(ddma_radar.angAxis_deg, ddma_radar.ULA_spectrumdB[ele,:],lw=2)
            #     plt.vlines(ddma_radar.objectAzAngle_deg[ele], ymin = -170, ymax = -110)
            #     plt.xlabel('Angle (deg)')
            #     plt.ylabel('dB')
            #     plt.grid(True)


    angleErrorMatrix_std[count_snrMC] = np.std(errorAngArray)
    angleErrorMatrix_percentile[count_snrMC] = np.percentile(np.abs(errorAngArray),percentile)


    angleSLLMatrix_max[count_snrMC] = np.amax(angleSllArray)
    angleSLLMatrix_median[count_snrMC] = np.median(angleSllArray)
    angleSLLMatrix_percentile[count_snrMC] = np.percentile(angleSllArray,percentile)


    count_snrMC += 1
    tstop_snr = time.time()
    timeSNR = tstop_snr - tstart_snr
    print('Time taken for {0}/{1} SNR run = {2:.2f} s'.format(count_snrMC, numSnrMC, timeSNR))

print('\n\n')

tstop = time.time()

timeMC = tstop - tstart
print('Total time for Monte-Carlo run = {0:.2f} min'.format(timeMC/60))

n = 1

plt.figure(n,figsize=(20,10), dpi=200)
plt.title('Angle Error(std) vs SNR. Number of DDMA chirps = ' + str(ddma_radar.numRamps))
plt.plot(range_binSNRArray, angleErrorMatrix_std, '-o')
plt.xlabel('SNR (dB)')
plt.ylabel('Angle Error std (deg)')
plt.grid(True)


n+=1

plt.figure(n,figsize=(20,10), dpi=200)
plt.title('Abs Angle Error(' + str(percentile) + ' percentile) vs SNR. Number of DDMA chirps = ' + str(ddma_radar.numRamps))
plt.plot(range_binSNRArray, angleErrorMatrix_percentile, '-o')
plt.xlabel('SNR (dB)')
plt.ylabel('deg')
plt.grid(True)

# plt.ylim([0,1])

""" Hanning window SLL"""
WindowFn = np.hanning(ddma_radar.numMIMO)
WindowFnFFT = np.fft.fft(WindowFn,n=ddma_radar.numAngleFFT)
WindowFnFFT = np.fft.fftshift(WindowFnFFT)
WindowFnFFTSpecMagdB = 20*np.log10(np.abs(WindowFnFFT))
WindowFnFFTSpecMagdBNorm = WindowFnFFTSpecMagdB - np.amax(WindowFnFFTSpecMagdB)

localMaxInd = argrelextrema(WindowFnFFTSpecMagdBNorm,np.greater,axis=0,order=2)[0]
sllInd = np.argsort(WindowFnFFTSpecMagdBNorm[localMaxInd])[-2]
WindSLL = WindowFnFFTSpecMagdBNorm[localMaxInd[sllInd]]


plt.figure(n+1,figsize=(20,10), dpi=200)
plt.title('Angle SLLs(dBc) vs SNR. Number of DDMA chirps = ' + str(ddma_radar.numRamps))
plt.plot(range_binSNRArray, angleSLLMatrix_max.T, '-o',label='Max SLL')
plt.plot(range_binSNRArray, angleSLLMatrix_median.T, '-o',label='Median SLL')
plt.plot(range_binSNRArray, angleSLLMatrix_percentile.T, '-o',label= str(percentile) + ' percentile SLL')
plt.axhline(WindSLL,color='k',label='Hanning Window SLL',linestyle='dashed')
plt.xlabel('SNR (dB)')
plt.ylabel('SLL (dBc)')
plt.grid(True)
plt.legend()
plt.ylim([-50,10])
n+=1


# """ Saving for plotting and debugging purposes"""
# isolation = tx0tx1IsolationPowerdB
# savepath = 'data_isolation\\withTxRxPhaseCoupling_AntennaCoupling_20dB_TxRx\\'
# np.save(savepath + 'angleErrorMatrix_std_isolation' + str(isolation) + 'dB.npy',angleErrorMatrix_std)
# np.save(savepath + 'angleErrorMatrix_percentile_isolation' + str(isolation) + 'dB.npy',angleErrorMatrix_percentile)
# np.save(savepath + 'range_binSNRArray_isolation' + str(isolation) + 'dB.npy',range_binSNRArray)
# np.save(savepath + 'angleSLLMatrix_median_isolation' + str(isolation) + 'dB.npy',angleSLLMatrix_median)




