# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 22:06:07 2023

@author: Sai Gunaranjan
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
from mimoPhasorSynthesis import mimoPhasorSynth # loading the antenna corodinates for the steradian RADAR platforms

# np.random.seed(10) # This is the true seed and not in the wrapper script

class DDMA_Radar:

    def __init__(self, flagEnableICCoupling, flagEnableAntennaCoupling, platform, flagRBM, \
                 phaseDemodMethod, flagEnableBoreSightCal):


        self.flagEnableICCoupling      = flagEnableICCoupling
        self.flagEnableAntennaCoupling = flagEnableAntennaCoupling
        self.platform                  = platform
        self.flagRBM                   = flagRBM
        self.flagEnableBoreSightCal    = flagEnableBoreSightCal
        self.phaseDemodMethod          = phaseDemodMethod


        self.numSamp = 2048 # Number of ADC time domain samples
        self.numSampPostRfft = self.numSamp//2
        self.numAngleFFT = 2048
        self.mimoArraySpacing = 2e-3 # 2mm
        self.lightSpeed = 3e8


        """ Chirp Parameters"""
        self.numDoppFFT = 2048
        self.chirpBW = 1e9 # Hz
        self.centerFreq = 76.5e9 # GHz
        self.interRampTime = 44e-6 # us
        self.chirpSamplingRate = 1/self.interRampTime
        self.rangeRes = self.lightSpeed/(2*self.chirpBW)
        self.maxRange = self.numSampPostRfft*self.rangeRes # m
        self.lamda = self.lightSpeed/self.centerFreq

        self.Fs_spatial = self.lamda/self.mimoArraySpacing
        self.angAxis_deg = np.arcsin(np.arange(-self.numAngleFFT//2, self.numAngleFFT//2)*(self.Fs_spatial/self.numAngleFFT))*180/np.pi

        """ RF parameters """
        self.thermalNoise = -174 # dBm/Hz
        self.noiseFigure = 10 # dB
        self.baseBandgain = 34 #dB
        self.adcSamplingRate = 56.25e6 # 56.25 MHz
        self.adcSamplingTime = 1/self.adcSamplingRate # s
        self.chirpOnTime = self.numSamp*self.adcSamplingTime
        self.chirpSlope = self.chirpBW/self.chirpOnTime
        self.dBFs_to_dBm = 10
        self.totalNoisePower_dBm = self.thermalNoise + self.noiseFigure + self.baseBandgain + 10*np.log10(self.adcSamplingRate)
        self.totalNoisePower_dBFs = self.totalNoisePower_dBm - 10
        self.noiseFloor_perBin = self.totalNoisePower_dBFs - 10*np.log10(self.numSamp) # dBFs/bin
        self.noisePower_perBin = 10**(self.noiseFloor_perBin/10)
        self.totalNoisePower = self.noisePower_perBin*self.numSamp # sigmasquare totalNoisePower
        self.sigma = np.sqrt(self.totalNoisePower)
        self.DoppAmbigNumArr = np.arange(-2,3) # Doppler Ambiguity number/Doppler Integer hypothesis


        if (self.flagRBM == 1):
            print('\n\nRange Bin Migration term has been enabled\n\n')

        if (self.flagEnableBoreSightCal == 1):
            print('\n\nBoresight cal has been enabled\n\n')
        else:
            print('\n\nBoresight cal has been disabled\n\n')

        if (self.phaseDemodMethod == 1):
            print('\n\nUsing the Tx demodulation method for DDMA\n\n')
        else:
            print('\n\nUsing modulated Doppler sampling method for DDMA\n\n')

        if ((self.flagEnableICCoupling == 1) and (self.flagEnableAntennaCoupling == 1) and (self.platform == 'SRIR16')):
            print('\n\nBoth IC and Antenna coupling enabled\n\n')
        elif  ((self.flagEnableICCoupling == 1) and (self.flagEnableAntennaCoupling == 0) and (self.platform == 'SRIR16')):
             print('\n\nIC coupling enabled but Antenna coupling disabled\n\n')
        elif  ((self.flagEnableICCoupling == 0) and (self.flagEnableAntennaCoupling == 1) and (self.platform == 'SRIR16')):
             print('\n\nIC coupling disabled but Antenna coupling enabled\n\n')
        elif ((self.flagEnableICCoupling == 0) and (self.flagEnableAntennaCoupling == 0) and (self.platform == 'SRIR16')):
            print('\n\nBoth IC and Antenna coupling disabled\n\n')
        else:
            print('\n\nInter Tx and Inter Rx coupling not supported for this platform currently\n\n')

        if (self.platform == 'SRIR16'):
            self.numTx_simult = 4
            self.numRx = 4
            self.numMIMO = 16 # All MIMO in azimuth only
            self.numRamps = 128 # Assuming 128 ramps for both detection and MIMO segments
            """ With 30 deg, we see periodicity since 30 divides 360 but with say 29 deg, it doesn't divide 360 and
            hence periodicity is significantly reduced"""
            self.phaseStepPerTx_deg = 89#29#29.3 This should ideally be 360/numTx_simult + delta, so that the Dopplers are spaced uniformly throughout the Doppler spectrum
        elif (self.platform == 'SRIR144'):
            self.numTx_simult = 12
            self.numRx = 12
            self.numMIMO = 48
            self.numRamps = 140 # Assuming 140 ramps for both detection and MIMO segments
            """ With 30 deg, we see periodicity since 30 divides 360 but with say 29 deg, it doesn't divide 360 and
            hence periodicity is significantly reduced"""
            self.phaseStepPerTx_deg = 29#29#29.3 This should ideally be 360/numTx_simult + delta, so that the Dopplers are spaced uniformly throughout the Doppler spectrum
        elif (self.platform == 'SRIR256'):
            self.numTx_simult = 13
            self.numRx = 16
            self.numMIMO = 74
            self.numRamps = 140 # Assuming 140 ramps for both detection and MIMO segments
            """ With 30 deg, we see periodicity since 30 divides 360 but with say 29 deg, it doesn't divide 360 and
            hence periodicity is significantly reduced"""
            self.phaseStepPerTx_deg = 29#29#29.3 This should ideally be 360/numTx_simult + delta, so that the Dopplers are spaced uniformly throughout the Doppler spectrum


        self.maxVelBaseband_mps = (self.chirpSamplingRate/2) * (self.lamda/2) # m/s
        self.FsEquivalentVelocity = 2*self.maxVelBaseband_mps # Fs = 2*Fs/2
        self.velocityRes = (self.chirpSamplingRate/self.numRamps) * (self.lamda/2)

        """ Derived Parameters """
        snrGainDDMA = 10*np.log10(self.numTx_simult**2) #dB
        snrGainDopplerFFT = 10*np.log10(self.numRamps) #dB
        self.totalsnrGain = snrGainDDMA + snrGainDopplerFFT



        return


    def introduce_coupling(self):

        """ Coupling introduces deterministic magnitude coupling across Txs and random phase contribution from adjacent Txs
        Diagonal elements of the phase coupling matrix are made 0. Is they are not zero, then it means we are introducing arbitrary phases
        for Txs and Rxs. In such case, bore-sight cal has to be enabled to remove these random phases.
        For now, we are introducing random phases in the signal generation function and hence not needed here. Hence diagonal elemants
        phase made to 0.
        """
        ## currently enabled only for single IC. Will add for multi IC later ON
        if (self.flagEnableICCoupling == 1) and (self.platform == 'SRIR16'):
            """ Typical Isolation/coupling numbers of Txs in a chip. Adjacent Txs have an isolation/coupling
            of about 20 dB and from there on it drops by about 6 dB as we move away from the Txs"""
            tx0tx1IsolationPowerdB = 20#20
            tx0tx2IsolationPowerdB = tx0tx1IsolationPowerdB + 6
            tx0tx3IsolationPowerdB = tx0tx2IsolationPowerdB + 6

            tx0tx1IsolationAmp = np.sqrt(1/(10**(tx0tx1IsolationPowerdB/10)))
            tx0tx2IsolationAmp = np.sqrt(1/(10**(tx0tx2IsolationPowerdB/10)))
            tx0tx3IsolationAmp = np.sqrt(1/(10**(tx0tx3IsolationPowerdB/10)))

            tx0tx0IsolationAmp = 1
            tx0tx1IsolationAmp = np.round(tx0tx1IsolationAmp,3)
            tx0tx2IsolationAmp = np.round(tx0tx2IsolationAmp,3)
            tx0tx3IsolationAmp = np.round(tx0tx3IsolationAmp,3)

            txisolationMagnitude = np.array([[tx0tx0IsolationAmp,tx0tx1IsolationAmp,tx0tx2IsolationAmp,tx0tx3IsolationAmp],\
                                                   [tx0tx1IsolationAmp,tx0tx0IsolationAmp,tx0tx1IsolationAmp,tx0tx2IsolationAmp],\
                                                       [tx0tx2IsolationAmp,tx0tx1IsolationAmp,tx0tx0IsolationAmp,tx0tx1IsolationAmp],\
                                                           [tx0tx3IsolationAmp,tx0tx2IsolationAmp,tx0tx1IsolationAmp,tx0tx0IsolationAmp]]) # These numbers correspond to power coupling of 20 dB, 20 + 6 dB, 20+6+6 dB and so on. More explanation given in docstring.


            txisolationPhase = 1*np.random.uniform(-np.pi,np.pi,self.numTx_simult*self.numTx_simult).reshape(self.numTx_simult,self.numTx_simult)
            txisolationPhasor = np.exp(1j*txisolationPhase)
            txisolationPhasor[np.arange(self.numTx_simult),np.arange(self.numTx_simult)] = 1
            self.txisolationMatrix = txisolationMagnitude*txisolationPhasor

            rxisolationMagnitude = txisolationMagnitude.copy()
            rxisolationPhase = 1*np.random.uniform(-np.pi,np.pi,self.numRx*self.numRx).reshape(self.numRx,self.numRx) # phase coupling from neighbouring Rxs
            rxisolationPhasor = np.exp(1j*rxisolationPhase)
            rxisolationPhasor[np.arange(self.numRx),np.arange(self.numRx)] = 1
            self.rxisolationMatrix = rxisolationMagnitude*rxisolationPhasor
        else:
            txisolationMagnitude = np.eye(self.numTx_simult)
            txisolationPhase = np.zeros((self.numTx_simult,self.numTx_simult))
            txisolationPhasor = np.exp(1j*txisolationPhase)
            self.txisolationMatrix = txisolationMagnitude*txisolationPhasor

            rxisolationMagnitude = np.eye(self.numRx)
            rxisolationPhase = np.zeros((self.numRx,self.numRx))
            rxisolationPhasor = np.exp(1j*rxisolationPhase)
            self.rxisolationMatrix = rxisolationMagnitude*rxisolationPhasor

        """ Typical radiated(at antenna) isolation/coupling numbers of Txs/Rx antennas. Adjacent Txs/Rxs antennas have an isolation/coupling
        of about 15 dB and from there on it drops by about 6 dB as we move away from the Txs.
        The antenna coupling/isolation is about 15 dB for lambda/2 separation and drops by 6 dB further on. Similarly,
        the antenna coupling/isolation is about 24 dB for 2 lambda"""
        if (self.flagEnableAntennaCoupling == 1) and (self.platform == 'SRIR16'):
            """ Tx antennas radiated coupling"""
            tx0tx1AntennaIsolationPowerdB = 15#15 # 15 or 20
            tx0tx2AntennaIsolationPowerdB = tx0tx1AntennaIsolationPowerdB + 6 # 20 with +12 or 15 with +6
            tx0tx3AntennaIsolationPowerdB = tx0tx2AntennaIsolationPowerdB + 6 # 20 with +12 or 15 with +6

            tx0tx1AntennaIsolationAmp = np.sqrt(1/(10**(tx0tx1AntennaIsolationPowerdB/10)))
            tx0tx2AntennaIsolationAmp = np.sqrt(1/(10**(tx0tx2AntennaIsolationPowerdB/10)))
            tx0tx3AntennaIsolationAmp = np.sqrt(1/(10**(tx0tx3AntennaIsolationPowerdB/10)))

            tx0tx0AntennaIsolationAmp = 1
            tx0tx1AntennaIsolationAmp = np.round(tx0tx1AntennaIsolationAmp,3)
            tx0tx2AntennaIsolationAmp = np.round(tx0tx2AntennaIsolationAmp,3)
            tx0tx3AntennaIsolationAmp = np.round(tx0tx3AntennaIsolationAmp,3)

            txAntennaisolationMagnitude = np.array([[tx0tx0AntennaIsolationAmp,tx0tx1AntennaIsolationAmp,tx0tx2AntennaIsolationAmp,tx0tx3AntennaIsolationAmp],\
                                                       [tx0tx1AntennaIsolationAmp,tx0tx0AntennaIsolationAmp,tx0tx1AntennaIsolationAmp,tx0tx2AntennaIsolationAmp],\
                                                           [tx0tx2AntennaIsolationAmp,tx0tx1AntennaIsolationAmp,tx0tx0AntennaIsolationAmp,tx0tx1AntennaIsolationAmp],\
                                                               [tx0tx3AntennaIsolationAmp,tx0tx2AntennaIsolationAmp,tx0tx1AntennaIsolationAmp,tx0tx0AntennaIsolationAmp]])


            txAntennaisolationPhase = 1*np.random.uniform(-np.pi,np.pi,self.numTx_simult*self.numTx_simult).reshape(self.numTx_simult,self.numTx_simult) # phase coupling from neighbouring Txs antennas
            txAntennaisolationPhasor = np.exp(1j*txAntennaisolationPhase)

            txAntennaisolationPhasor[np.arange(self.numTx_simult),np.arange(self.numTx_simult)] = 1
            self.txAntennaisolationMatrix = txAntennaisolationMagnitude*txAntennaisolationPhasor

            """ Rx antennas radiated coupling"""
            rx0rx1AntennaIsolationPowerdB = 24#24 # 20 or 24
            rx0rx2AntennaIsolationPowerdB = rx0rx1AntennaIsolationPowerdB + 6 # 20 with +12 or 24 with +6
            rx0rx3AntennaIsolationPowerdB = rx0rx2AntennaIsolationPowerdB + 6 # 20 with +12 or 24 with +6

            rx0rx1AntennaIsolationAmp = np.sqrt(1/(10**(rx0rx1AntennaIsolationPowerdB/10)))
            rx0rx2AntennaIsolationAmp = np.sqrt(1/(10**(rx0rx2AntennaIsolationPowerdB/10)))
            rx0rx3AntennaIsolationAmp = np.sqrt(1/(10**(rx0rx3AntennaIsolationPowerdB/10)))

            rx0rx0AntennaIsolationAmp = 1
            rx0rx1AntennaIsolationAmp = np.round(rx0rx1AntennaIsolationAmp,3)
            rx0rx2AntennaIsolationAmp = np.round(rx0rx2AntennaIsolationAmp,3)
            rx0rx3AntennaIsolationAmp = np.round(rx0rx3AntennaIsolationAmp,3)

            rxAntennaisolationMagnitude = np.array([[rx0rx0AntennaIsolationAmp,rx0rx1AntennaIsolationAmp,rx0rx2AntennaIsolationAmp,rx0rx3AntennaIsolationAmp],\
                                                       [rx0rx1AntennaIsolationAmp,rx0rx0AntennaIsolationAmp,rx0rx1AntennaIsolationAmp,rx0rx2AntennaIsolationAmp],\
                                                           [rx0rx2AntennaIsolationAmp,rx0rx1AntennaIsolationAmp,rx0rx0AntennaIsolationAmp,rx0rx1AntennaIsolationAmp],\
                                                               [rx0rx3AntennaIsolationAmp,rx0rx2AntennaIsolationAmp,rx0rx1AntennaIsolationAmp,rx0rx0AntennaIsolationAmp]])


            rxAntennaisolationPhase = 1*np.random.uniform(-np.pi,np.pi,self.numRx*self.numRx).reshape(self.numRx,self.numRx) # phase coupling from neighbouring Rxs
            rxAntennaisolationPhasor = np.exp(1j*rxAntennaisolationPhase)

            rxAntennaisolationPhasor[np.arange(self.numTx_simult),np.arange(self.numTx_simult)] = 1
            self.rxAntennaisolationMatrix = rxAntennaisolationMagnitude*rxAntennaisolationPhasor

        else:
            txAntennaisolationMagnitude = np.eye(self.numTx_simult)
            txAntennaisolationPhase = np.zeros((self.numTx_simult,self.numTx_simult))
            txAntennaisolationPhasor = np.exp(1j*txAntennaisolationPhase)
            self.txAntennaisolationMatrix = txAntennaisolationMagnitude*txAntennaisolationPhasor

            rxAntennaisolationMagnitude = np.eye(self.numRx)
            rxAntennaisolationPhase = np.zeros((self.numRx,self.numRx))
            rxAntennaisolationPhasor = np.exp(1j*rxAntennaisolationPhase)
            self.rxAntennaisolationMatrix = rxAntennaisolationMagnitude*rxAntennaisolationPhasor

        return


    def print_system_info(self):


        print('\n\nTotal SNR gain ( {0:.0f} Tx DDMA + {1:.0f} point Doppler FFT) = {2:.2f} dB'.format(self.numTx_simult, self.numRamps, self.totalsnrGain))

        print('Max base band velocity = {0:.2f} m/s'.format(self.maxVelBaseband_mps))

        print('Velocity resolution = {0:.2f} m/s'.format(self.velocityRes))

        print('Noise Floor set by DNL: {} dB'.format(np.round(self.noiseFloorSetByDNL)))

        return


    def define_phaseShifter_settings(self):

        """ Phase Shifter settings"""
        self.numBitsPhaseShifter = 7
        self.numPhaseCodes = 2**(self.numBitsPhaseShifter)
        self.DNL = 360/(self.numPhaseCodes) # DNL in degrees

        self.phaseStepPerRamp_deg = np.arange(self.numTx_simult)*self.phaseStepPerTx_deg # Phase step per ramp per Tx
        self.phaseStepPerRamp_rad = (self.phaseStepPerRamp_deg/360)*2*np.pi
        phaseShifterCodes = self.DNL*np.arange(self.numPhaseCodes)
        phaseShifterNoise = np.random.uniform(-self.DNL/2, self.DNL/2, self.numPhaseCodes)
        self.phaseShifterCodes_withNoise = phaseShifterCodes + phaseShifterNoise
        """ Ensure that the phase shifter LUT is without any bias and is from 0 to 360 degrees"""
        self.phaseShifterCodes_withNoise = np.mod(self.phaseShifterCodes_withNoise,360)

        DNL_rad = (self.DNL/180) * np.pi
        self.noiseFloorSetByDNL = 10*np.log10((DNL_rad)**2/12) - 10*np.log10(self.numRamps) + 10*np.log10(self.numTx_simult) # DNL Noise floor raises as 10log10(numSimulTx)



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
        self.rampPhaseIdeal_deg = self.phaseStepPerRamp_deg[:,None]*(np.arange(self.numRamps)[None,:] + (alpha/2)*(np.arange(self.numRamps)[None,:])**2)
        self.rampPhaseIdeal_degWrapped = np.mod(self.rampPhaseIdeal_deg, 360)
        self.phaseCodesIndexToBeApplied = np.argmin(np.abs(self.rampPhaseIdeal_degWrapped[:,:,None] - self.phaseShifterCodes_withNoise[None,None,:]),axis=2)
        self.phaseCodesToBeApplied = self.phaseShifterCodes_withNoise[self.phaseCodesIndexToBeApplied]
        self.phaseCodesToBeApplied_rad = (self.phaseCodesToBeApplied/180) * np.pi

        return


    def target_definitions(self):

        """ Target definition"""
        self.objectRange = np.random.uniform(10,self.maxRange-10) # 60.3 # m
        self.objectRangeBin = self.objectRange/self.rangeRes

        self.numDopUniqRbin = int(np.random.randint(low=1, high=4, size=1)) #2 # Number of Dopplers in a given range bin

        # objectVelocity_mps = np.random.uniform(-maxVelBaseband_mps-2*FsEquivalentVelocity, maxVelBaseband_mps+2*FsEquivalentVelocity, numDopUniqRbin)  #np.array([-10,-10.1]) #np.array([-10,23])# m/s

        # self.objectVelocity_mps = np.random.uniform(-self.maxVelBaseband_mps+(self.DoppAmbNum*self.FsEquivalentVelocity), \
        #                                         -self.maxVelBaseband_mps+(self.DoppAmbNum*self.FsEquivalentVelocity)+self.FsEquivalentVelocity,self.numDopUniqRbin)

        self.objectVelocity_mps = np.empty([0])
        for numVels in np.arange(self.numDopUniqRbin):
            """ -1/+1 hypothesis is 3 times as likely as -2/2 hypothesis. 0 hypthesis is 2 times as likely as -1/+1 hypothesis """
            DoppAmbNum = np.random.choice(self.DoppAmbigNumArr,p=[1/20, 3/20, 12/20, 3/20, 1/20])
            speedEachTarget = np.random.uniform(-self.maxVelBaseband_mps+(DoppAmbNum*self.FsEquivalentVelocity), \
                                        -self.maxVelBaseband_mps+(DoppAmbNum*self.FsEquivalentVelocity)+self.FsEquivalentVelocity,1)
            self.objectVelocity_mps = np.append(self.objectVelocity_mps,speedEachTarget)

        # print('Velocities (mps):', np.round(self.objectVelocity_mps,2))
        self.objectVelocity_baseBand_mps = np.mod(self.objectVelocity_mps, self.FsEquivalentVelocity) # modulo Fs [from 0 to Fs]
        self.objectVelocityBin = self.objectVelocity_baseBand_mps/self.velocityRes
        self.objectVelocity_baseBand_mpsBipolar = self.objectVelocity_baseBand_mps
        self.objectVelocity_baseBand_mpsBipolar[self.objectVelocity_baseBand_mpsBipolar>=self.FsEquivalentVelocity/2] -= self.FsEquivalentVelocity

        self.objectAzAngle_deg = np.random.uniform(-50,50, self.numDopUniqRbin) #np.array([30,-10]) Theta plane angle
        objectAzAngle_rad = (self.objectAzAngle_deg/360) * (2*np.pi)

        objectElAngle_deg = np.zeros((self.numDopUniqRbin,)) # phi=0 plane angle
        objectElAngle_rad = (objectElAngle_deg/360) * (2*np.pi)

        mimoPhasor, self.mimoPhasor_txrx, self.ulaInd = mimoPhasorSynth(self.platform, self.lamda, objectAzAngle_rad, objectElAngle_rad)

        """ Cal target settings"""
        calTargetRange = 5 # in m
        self.calTargetRangeBin = np.round(calTargetRange/self.rangeRes).astype('int32')
        self.calTargetRange = self.calTargetRangeBin*self.rangeRes # To ensure cal target falls exactly on a range bin
        self.calTargetVelocityBin = np.array([0])
        self.calTargetAzAngle_deg = np.array([0])
        self.calTargetElAngle_deg = np.array([0]) # phi=0 plane angle

        return


    def ddma_signal_generation(self, binSNR):

        signalPowerdBFs = self.noiseFloor_perBin + binSNR
        signalPower = 10**(signalPowerdBFs/10)
        signalAmplitude = np.sqrt(signalPower)
        signalPhase = np.exp(1j*np.random.uniform(-np.pi, np.pi))
        signalphasor = signalAmplitude*signalPhase

        self.rangeTerm = signalphasor*np.exp(1j*((2*np.pi*self.objectRangeBin)/self.numSamp)*np.arange(self.numSamp))
        self.dopplerTerm = np.exp(1j*((2*np.pi*self.objectVelocityBin[:,None])/self.numRamps)*np.arange(self.numRamps)[None,:]) # [number of Dopplers/range, numRamps]
        """ Range Bin migration term"""
        self.rangeBinMigration = \
            np.exp(1j*2*np.pi*self.chirpSlope*(2*self.objectVelocity_mps[:,None,None]/self.lightSpeed)*self.interRampTime*self.adcSamplingTime*np.arange(self.numRamps)[None,:,None]*np.arange(self.numSamp)[None,None, :])

        self.rxSignal = self.mimoPhasor_txrx[:,0,:]
        self.txSignal = self.mimoPhasor_txrx[:,:,0]

        self.unCaliberatedTxPhasesRad = np.random.uniform(-np.pi,np.pi, self.numTx_simult)
        self.unCaliberatedTxPhasor = np.exp(1j*self.unCaliberatedTxPhasesRad)

        self.signal_phaseCode = np.exp(1j*self.phaseCodesToBeApplied_rad)
        self.signal_phaseCode_txPhaseDisturbed = self.signal_phaseCode * self.unCaliberatedTxPhasor[:,None]
        signal_phaseCode_couplingMatrix = self.txAntennaisolationMatrix @ self.txisolationMatrix @ self.signal_phaseCode_txPhaseDisturbed
        txWeights = np.ones((self.numTx_simult,),dtype=np.float32) #np.array([1,1,1,1])# amplitide varation across Txs. Currently assuming all Txs have same gain
        self.signal_phaseCode_couplingMatrix_txWeights = txWeights[:,None]*signal_phaseCode_couplingMatrix


        phaseCodedTxSignal = self.dopplerTerm[:,None,:] * self.signal_phaseCode_couplingMatrix_txWeights[None,:,:] * self.txSignal[:,:,None] # [numDopp, numTx, numRamps]
        phaseCodedTxRxSignal = phaseCodedTxSignal[:,:,:,None]*self.rxSignal[:,None,None,:] #[numDopp, numTx, numRamps, numTx, numRx]
        phaseCodedTxRxSignal_withRangeTerm = self.rangeTerm[None,None,None,None,:] * phaseCodedTxRxSignal[:,:,:,:,None]
        if (self.flagRBM == 1):
            phaseCodedTxRxSignal_withRangeTerm = phaseCodedTxRxSignal_withRangeTerm * self.rangeBinMigration[:,None,:,None,:]


        signal = np.sum(phaseCodedTxRxSignal_withRangeTerm, axis=(0,1)) # [numRamps,numRx, numSamp]
        """ Construct cal target"""
        self.ddma_calsignal_construction()

        signal += self.calTargetsignal # Adding the cal taregt signal to the actual signal

        """ Rx coupling"""
        """ Right now introducing Rx IC isolation followed by Rx antenna isolation and then adding noise.
        This is ideally not the correct way.
        First need to introduce Rx antenna coupling followed by noise addition and then Rx IC coupling.
        Need to make this change"""

        """ Also need to introduce arbitrary phases in Rxs and let the bore-sight caliberation handle this."""
        signal = np.matmul(self.rxisolationMatrix[None,:,:],signal) # Tx IC coupling x Rx signal
        signal = np.matmul(self.rxAntennaisolationMatrix[None,:,:],signal) # Rx Antenna coupling x Rx IC coupling x Rx signal

        noise = (self.sigma/np.sqrt(2))*np.random.randn(self.numRamps*self.numRx*self.numSamp) + 1j*(self.sigma/np.sqrt(2))*np.random.randn(self.numRamps*self.numRx*self.numSamp)
        noise = noise.reshape(self.numRamps, self.numRx, self.numSamp)
        self.signal = signal + noise

        # """ Rx coupling"""
        # """ Right now introducing Rx IC isolation followed by Rx antenna isolation and then adding noise.
        # This is ideally not the correct way.

        # First need to introduce Rx antenna coupling followed by noise addition and then Rx IC coupling.
        # Need to make this change"""

        # """ Also need to introduce arbitrary phases in Rxs and let the bore-sight caliberation handle this."""
        # """ 1. Rx Antenna coupling to the received signal
        #     2. Add noise to the received signal at LNA
        #     3. Rx IC coupling to the LNA signal (post addition of noise)
        # """
        # """ 1. Rx antenna coupling"""
        # signal = np.matmul(self.rxAntennaisolationMatrix[None,:,:],signal) # Rx Antenna coupling x Rx signal
        # """ 2. Add noise"""
        # noise = (self.sigma/np.sqrt(2))*np.random.randn(self.numRamps*self.numRx*self.numSamp) + 1j*(self.sigma/np.sqrt(2))*np.random.randn(self.numRamps*self.numRx*self.numSamp)
        # noise = noise.reshape(self.numRamps, self.numRx, self.numSamp)
        # signal = signal + noise
        # """ 3. Rx IC coupling"""
        # self.signal = np.matmul(self.rxisolationMatrix[None,:,:],signal) # Rx IC coupling x  Rx Antenna coupling x Rx signal


        return


    def ddma_calsignal_construction(self):

        if (self.flagEnableBoreSightCal == 1):
            calrangeTerm = 0.125*np.exp(1j*((2*np.pi*self.calTargetRangeBin)/self.numSamp)*np.arange(self.numSamp))
            caldopplerTerm = np.exp(1j*((2*np.pi*self.calTargetVelocityBin[:,None])/self.numRamps)*np.arange(self.numRamps)[None,:]) # [number of Dopplers/range, numRamps]
            calTargetAzAngle_rad = (self.calTargetAzAngle_deg/360) * (2*np.pi)
            calTargetElAngle_rad = (self.calTargetElAngle_deg/360) * (2*np.pi)
            _, mimoPhasor_txrx_caltarget, _ = mimoPhasorSynth(self.platform, self.lamda, calTargetAzAngle_rad, calTargetElAngle_rad)
            caltargetrxSignal = mimoPhasor_txrx_caltarget[:,0,:]
            calTargettxSignal = mimoPhasor_txrx_caltarget[:,:,0]

            calTargetphaseCodedTxSignal = caldopplerTerm[:,None,:] * self.signal_phaseCode_couplingMatrix_txWeights[None,:,:] * calTargettxSignal[:,:,None] # [numDopp, numTx, numRamps]
            calTargetphaseCodedTxRxSignal = calTargetphaseCodedTxSignal[:,:,:,None]*caltargetrxSignal[:,None,None,:] #[numDopp, numTx, numRamps, numTx, numRx]
            calTargetphaseCodedTxRxSignal_withRangeTerm = calrangeTerm[None,None,None,None,:] * calTargetphaseCodedTxRxSignal[:,:,:,:,None]
            self.calTargetsignal = np.sum(calTargetphaseCodedTxRxSignal_withRangeTerm, axis=(0,1)) # [numRamps,numRx, numSamp]
        else:
            self.calTargetsignal = 0


        return

    def ddma_range_processing(self):


        signal_rangeWin = self.signal*np.hanning(self.numSamp)[None,None,:]
        self.signal_rfft = np.fft.fft(signal_rangeWin,axis=2)/self.numSamp
        self.signal_rfft = self.signal_rfft[:,:,0:self.numSampPostRfft]
        self.signal_rfft_powermean = np.mean(np.abs(self.signal_rfft)**2,axis=(0,1))

        if (self.flagRBM == 1):
            rangeMoved = self.objectRange + self.objectVelocity_mps[:,None]*self.interRampTime*np.arange(self.numRamps)[None,:]
        else:
            rangeMoved = self.objectRange + 0*self.objectVelocity_mps[:,None]*self.interRampTime*np.arange(self.numRamps)[None,:]
        rangeBinsMovedfrac = rangeMoved/self.rangeRes + 0*np.random.uniform(-0.5,0.5,self.numDopUniqRbin)[:,None]
        self.rangeBinsToSample = np.floor(rangeBinsMovedfrac).astype('int32')
        self.chirpSamp_givenRangeBin = self.signal_rfft[np.arange(self.numRamps)[None,:],:,self.rangeBinsToSample]

        return

    def rbm_phase_freqbin_correction(self):

        if (self.flagRBM == 1):
            """ Correcting for the Pi phase jump caused due to the Range bin Migration"""
            binDelta = np.abs(self.rangeBinsToSample[:,1::] - self.rangeBinsToSample[:,0:-1])
            tempVar = binDelta*np.pi #tempVar = binDelta*(np.pi - np.pi/self.numSamp)
            binMigrationPhaseCorrTerm = np.zeros((self.rangeBinsToSample.shape),dtype=np.float32)
            binMigrationPhaseCorrTerm[:,1::] = tempVar
            binMigrationPhasorCorrTerm = np.exp(-1j*np.cumsum(binMigrationPhaseCorrTerm,axis=1))
            self.chirpSamp_givenRangeBin = self.chirpSamp_givenRangeBin*binMigrationPhasorCorrTerm[:,:,None]

            """ Correcting for the Doppler modulation caused due to the Range bin Migration"""

            """ To correct for the Doppler modulation caused due to range migration, we could do it in 2 ways:
                1. Compensate the Doppler modulation as a phase correction the chirp/Doppler samples
                2. Convert the Doppler modulation to a bin and add this bin to the true Doppler bin. So in essence,
                we are converting the probelm of phase correction/demodulation to updated Doppler bin sampling

                Currently, I'm using method 2'
            """
            rbmModulationAnalogFreq = (self.chirpBW/self.lightSpeed)*self.objectVelocity_mps
            self.dopplerBinOffset_rbm = (rbmModulationAnalogFreq/self.chirpSamplingRate)*self.numDoppFFT

            # rbmModulationDigitalFreq = (rbmModulationAnalogFreq/chirpSamplingRate)*2*np.pi
            # rbmModulationCorrectionTerm = np.exp(-1j*rbmModulationDigitalFreq[:,None]*np.arange(numRamps)[None,:])
            # self.chirpSamp_givenRangeBin = self.chirpSamp_givenRangeBin*rbmModulationCorrectionTerm[:,:,None]
        else:
            self.dopplerBinOffset_rbm = np.zeros((self.numDopUniqRbin,))

        return


    def mimo_coefficient_estimation(self):

        objectVelocityBinNewScale = (self.objectVelocityBin/self.numRamps)*self.numDoppFFT
        self.binOffset_Txphase = (self.phaseStepPerRamp_rad/(2*np.pi))*self.numDoppFFT

        """ To obtain the signal coefficients, we could either do an FFT and evaluate at the required bins
        or we could simply perform a single point DFT by correlating the signal with the phasor constructed out of the required bins.
        In this code, I'm evaluating the coefficients by doing a single point DFT at the known Doppler bins.
        This way, we don't have to perform a huge FFT and then evaluate at the dopplerBinsToSample which is compute heavy.
        Rather, we evaluate a single point DFT at the dopplerBinsToSample.
        This has a much smaller compute as compared to performing a huge FFT for 16 Rx channels.
        Here, I'm computing the Doppler FFT for plotting purpose only"""

        if (self.phaseDemodMethod == 1):
            self.dopplerBinsToSample = np.round(objectVelocityBinNewScale[:,None] + self.dopplerBinOffset_rbm[:,None]).astype('int32')
            self.dopplerBinsToSample = np.repeat(self.dopplerBinsToSample,self.numTx_simult,axis=1)
            demodulatingPhaseCodedSignal = self.signal_phaseCode.T # Transpose is to just matching the dimensions of the chirp samples signal
            demodulatedDopplerSignal = self.chirpSamp_givenRangeBin[:,:,:,None]*np.conj(demodulatingPhaseCodedSignal)[None,:,None,:] # [numDopp, numChirp, numRx, numTx]
            signalWindowed = demodulatedDopplerSignal*np.hanning(self.numRamps)[None,:,None,None] # [numDopp, numChirp, numRx, numTx]
            """DFT based coefficient estimation """
            DFT_vec = np.exp(1j*2*np.pi*(self.dopplerBinsToSample[:,None,:]/self.numDoppFFT)*np.arange(self.numRamps)[None,:,None])
            mimoCoefficients_eachDoppler_givenRange = np.sum(signalWindowed*np.conj(DFT_vec[:,:,None,:]),axis=1)/self.numRamps
            self.mimoCoefficients_eachDoppler_givenRange = np.transpose(mimoCoefficients_eachDoppler_givenRange,(0,2,1))

            self.signalFFT = np.fft.fft(signalWindowed, axis=1, n = self.numDoppFFT)/self.numRamps
            """ Coefficients estimated using FFT and sampling the required bins"""
            # mimoCoefficients_eachDoppler_givenRange = self.signalFFT[np.arange(numDopUniqRbin),self.dopplerBinsToSample[:,0],:,:] # [numObj, numRx, numTx] , dopplerBinsToSample[:,0] because the other columns are basically repeats
            # self.mimoCoefficients_eachDoppler_givenRange = np.transpose(mimoCoefficients_eachDoppler_givenRange,(0,2,1)) # [numObj, numTx, numRx]
        else:

            self.dopplerBinsToSample = np.round(objectVelocityBinNewScale[:,None] + self.dopplerBinOffset_rbm[:,None] + self.binOffset_Txphase[None,:]).astype('int32')
            self.dopplerBinsToSample = np.mod(self.dopplerBinsToSample, self.numDoppFFT)
            signalWindowed = self.chirpSamp_givenRangeBin*np.hanning(self.numRamps)[None,:,None]
            """DFT based coefficient estimation """
            DFT_vec = np.exp(1j*2*np.pi*(self.dopplerBinsToSample[:,None,:]/self.numDoppFFT)*np.arange(self.numRamps)[None,:,None])
            mimoCoefficients_eachDoppler_givenRange = np.sum(signalWindowed[:,:,:,None]*np.conj(DFT_vec[:,:,None,:]),axis=1)/self.numRamps
            self.mimoCoefficients_eachDoppler_givenRange = np.transpose(mimoCoefficients_eachDoppler_givenRange,(0,2,1))

            self.signalFFT = np.fft.fft(signalWindowed, axis=1, n = self.numDoppFFT)/self.numRamps
            """ Coefficients estimated using FFT and sampling the required bins"""
            # self.mimoCoefficients_eachDoppler_givenRange = self.signalFFT[np.arange(numDopUniqRbin)[:,None],self.dopplerBinsToSample,:] # [numObj, numTx, numRx]

        return

    def extract_boresight_cal(self):

        if (self.flagEnableBoreSightCal == 1):
            if (self.phaseDemodMethod == 1):
                calTargetrangeBinsToSample = np.repeat(self.calTargetRangeBin,self.numRamps)
                calTargetVelocityBinsToSample = np.round(self.calTargetVelocityBin[:,None] + self.dopplerBinOffset_rbm[:,None]).astype('int32')
                calTargetVelocityBinsToSample = np.repeat(calTargetVelocityBinsToSample,self.numTx_simult,axis=1)
                demodulatingPhaseCodedSignal = self.signal_phaseCode.T # Transpose is to just matching the dimensions of the chirp samples signal
                chirpSamp_calTargetRangeBin = self.signal_rfft[np.arange(self.numRamps)[None,:],:,calTargetrangeBinsToSample]
                demodulatedDopplerSignalCaltarget = chirpSamp_calTargetRangeBin[:,:,:,None]*np.conj(demodulatingPhaseCodedSignal)[None,:,None,:] # [numDopp, numChirp, numRx, numTx]
                calTargetsignalWindowed = demodulatedDopplerSignalCaltarget#*np.hanning(numRamps)[None,:,None,None] # [numDopp, numChirp, numRx, numTx]
                """DFT based coefficient estimation """
                calTargetDFT_vec = np.exp(1j*2*np.pi*(calTargetVelocityBinsToSample[:,None,:]/self.numDoppFFT)*np.arange(self.numRamps)[None,:,None])
                mimoCoefficientsCalTarget = np.sum(calTargetsignalWindowed*np.conj(calTargetDFT_vec[:,:,None,:]),axis=1)/self.numRamps
                mimoCoefficientsCalTarget = np.transpose(mimoCoefficientsCalTarget,(0,2,1))

                mimoCoefficientsCalTarget_flatten = mimoCoefficientsCalTarget.reshape(-1, self.numTx_simult*self.numRx)
                mimoCoefficientsCalTarget_flatten = mimoCoefficientsCalTarget_flatten[:,self.ulaInd]
                self.calCoeffs = np.conj(mimoCoefficientsCalTarget_flatten/np.abs(mimoCoefficientsCalTarget_flatten))

            else:
                calTargetrangeBinsToSample = np.repeat(self.calTargetRangeBin,self.numRamps)
                chirpSamp_calTargetRangeBin = self.signal_rfft[np.arange(self.numRamps)[None,:],:,calTargetrangeBinsToSample]
                calTargetsignalWindowed = chirpSamp_calTargetRangeBin#*np.hanning(numRamps)[None,:,None]
                calTargetVelocityBinsToSample = np.round(self.calTargetVelocityBin[:,None] + self.binOffset_Txphase[None,:]).astype('int32')
                calTargetDFT_vec = np.exp(1j*2*np.pi*(calTargetVelocityBinsToSample[:,None,:]/self.numDoppFFT)*np.arange(self.numRamps)[None,:,None])

                mimoCoefficientsCalTarget = np.sum(calTargetsignalWindowed[:,:,:,None]*np.conj(calTargetDFT_vec[:,:,None,:]),axis=1)/self.numRamps
                mimoCoefficientsCalTarget = np.transpose(mimoCoefficientsCalTarget,(0,2,1))

                mimoCoefficientsCalTarget_flatten = mimoCoefficientsCalTarget.reshape(-1, self.numTx_simult*self.numRx)
                mimoCoefficientsCalTarget_flatten = mimoCoefficientsCalTarget_flatten[:,self.ulaInd]
                self.calCoeffs = np.conj(mimoCoefficientsCalTarget_flatten/np.abs(mimoCoefficientsCalTarget_flatten))

        else:
            self.calCoeffs = 1

        return


    def angle_estimation(self):

        mimoCoefficients_flatten = self.mimoCoefficients_eachDoppler_givenRange.reshape(-1, self.numTx_simult*self.numRx)
        mimoCoefficients_flatten = mimoCoefficients_flatten[:,self.ulaInd]
        mimoCoefficients_flattenCal = mimoCoefficients_flatten*self.calCoeffs
        self.ULA = np.unwrap(np.angle(mimoCoefficients_flattenCal),axis=1)
        mimoCoefficients_flattenCal = mimoCoefficients_flattenCal*np.hanning(self.numMIMO)[None,:]
        ULA_spectrum = np.fft.fft(mimoCoefficients_flattenCal,axis=1,n=self.numAngleFFT)/(self.numMIMO)
        ULA_spectrum = np.fft.fftshift(ULA_spectrum,axes=(1,))
        self.ULA_spectrumdB = 20*np.log10(np.abs(ULA_spectrum))
        self.ULA_spectrumdB -= np.amax(self.ULA_spectrumdB,axis=1)[:,None]

        return


