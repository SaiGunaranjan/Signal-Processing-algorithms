# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 17:44:19 2022

@author: Sai Gunaranjan
"""

"""
In this script, I will be modeling the FMCW + DAC-based transmission at the RF level.
The baseband behavior in an FMCW + DAC transmission is different from a pure FMCW transmission.
I will be highlighting these differences. Also, I will try and study the phase shifter behavior in this mode.

In FMCW RADAR architecture, we transmit a linear chirp signal with centre frequency around 77GHz.
Sometimes, for internal caliberation purposes and phase noise measurements,
an internal loopback mode is enabled in the hardware RFIC. In other words, instead of a radiated measurement,
the RF signal is sent over an internal metallic connection within the IC at the RF front end.
This is called the internal loopback transmission.
We can have different types of internal loopback mechanisms namely baseband loopback, PLL loopback, DAC loopback.

PLL loopback
Typically, an FMCW RADAR IC has (atleast)2 PLLs namely PLL1 and PLL2.
The PLL1 is responsible for generating the chirp signal/local oscillator(LO) signal at RF front end.
The PLL2 is responsible for generating the ADC clock and other clock driven circuits at the baseband.
The PLL2 loopback is used for internal phase noise measurements. In the PLL2 loopback method,
the Tx signal is generated by PLL2 and the mixing LO signal is generated by PLL1.
The Tx signal (generated by PLL2) is sent over an internal metallic loopback path to the Rx chain
where the signal is mixed with the LO signal (generated by PLL1). Since, there are 2 different PLLs involved,
there is no cancellation of phase noise and hance we can measure the phase noise.

DAC loopback
In the DAC loopback method, the PLL1 generates the chirp signal. Then, a Digital to Analog converter(DAC)
is used to generate a sinusoidal signal of known frequency and amplitude.
This DAC tone should also have a DC(in order for the FMCW tone to appear in baseband.
I will show this in a separate derivation). Now, the chirp signal is modulated with this DAC signal.
This signal now is transmitted both into free space using the antennas as well as
through an internal loopback metallic connection from Tx to Rx. At the receiver end,
the free space reflected signal as well as the loopback signal are demodulated using the clean LO.
Since both the transmitted signal as well as the LO signal are generated by PLL1,
the phase noise of the TX signal gets cancelled out by the phase noise of the LO (since both are generated using PLL1).
Hence, this mode of internal loopback doesn't suffer from any phase noise (unlike the PLL2 loopback mode).
This is the FMCW+DAC loopback mechanism. The DAC tone in our RADAR architecture is real.
The DAC tone should have a DC for the target frequencies(targets in open space) to appear. Similarly,
there should be a 0 delay path or internal loopback (which essentially translates to DC in the baseband spectrum)
for the DAC tone to appear in the baseband ADC spectrum. This point is very important.
Another important point to note is that in this simulation model, in addition to the target frequencies,
we obtain the (DAC + target frequencies) and -(DAC-target frequencies).
In other words, we don't observe the DAC-target frequencies and -(DAC+target frequencies).
This is because, the DAC signal is real, while the target signal is complex.
Further explanation: Since we have positive ramps(positive slope) in the FMCW,
we will obtain the target frequencies as positive. Now these positive frequencies get added/modulated by the DAC signal
which has both positive and negative frequencies (since the DAC signal is real).
Ex: if the DAC signal has frequencies f0 and -f0, and if the taregt frequencies are f1, f2, ..., then we will obtain
f0+f1, fo+f2, .... and -f0+f1, -f0+f2, ... and so on. This is in addition to the actual target frequencies
f1, f2, .. which are obatined due to the FMCW path. Hence, in the FMCW + DAC mode,
1. Target tones(only positive side) appear due to the FMCW mode.
2. DAC tones (both positive and negative since DAC signal is real) appear due to 0 delay/internal loopback path.
3. DAC + target frequencies , -DAC + target frequencies due to DAC signal being real and target signal being complex.


Simulation environment:
Coming to the simulation aspect of this model, I have made the following asumptions.
1. Generation of the chirp signal is done at 10GHz instead of 77GHz. This is because, if I simulate at 77GHz,
then the sampling rate for the chirp signal must be atleast 7-8 times of 77GHz.
This will lead to huge number of chirp samples (prior to ADC sampling).
The memory required for the variables will shoot up and cause the script to crash.
Hence, instead of simulating the chirp signal at 77GHz, I simulate at 10 GHz.
2. Ideally the analog signal transmitted into free space is a real signal (real chirp signal).
However, for simplicity sake, I have modelled the chirp signal/LO signal as a complex chirp signal.
When the LO signal is modelled as a complex signal, then there is no need for an anti aliasing filter since,
we will not obtain the high freqeuncy components due to mixing. But if the LO signal is real,
then we will obtain both the difference as well as sum of frequencies(high pass components) and
hence before ADC sampling, there is a need for an anti alaisign filter.
In the subsequent commits, I will change the LO signal also to be a real signal and
will also introduce the anti aliasing filter before ADC sampling.
3. Another important point to keep in mind while modelling is the oversampling rate (OSR).
The OSR must be chosen such that, this factor is same for the ADC sampling rate to RF sampling rate conversion
as well as number of ADC samples to number of RF samples. This can be achieved by setting the inter-chirp time
as an integer multiple of the ADC sampling time. The inter-chirp time cannot be set arbitrarily.
Else, we will see that the peak bins from the baseband spectrum pre-ADC sampling will not match
the peak bins of the baseband spectrum post ADC sampling.
For them to match, ratio Fs1/N1 = Fs2/N2, where Fs1 is the sampling rate for the RF signal and
N1 is the number of baseband samples pre ADc sampling.
Fs2 is the ADC sampling rate and N2 is the number of samples in the chirp post ADC sampling.
The ADC samples are obatined by simply downsampling the baseband signal by the OSR.
There is no need for anti-aliasing filter before downsampling since there are no high frequencies.

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.close('all')
lightSpeed = 3e8 #mps
initial_phase_deg = 0#30
initial_phase_rad = initial_phase_deg*np.pi/180
numChirpSamples = 2048
adcSamplingRate = 56.25e6 #Hz
chirp_time = numChirpSamples/adcSamplingRate# 36.4e-6 # us
chirpBW = 2e9#250e6#2e9
slope = chirpBW/chirp_time #25e6/1e-6 # MHz/us
Fstart_hz = 10e9 # GHz
rangeRes = lightSpeed/(2*chirpBW) #m

numPhaseCodes = 40
phaseShifter = np.linspace(0,360,numPhaseCodes)
phaseShifterRad = (phaseShifter/180) * np.pi
phaseShifterPhasor = np.exp(1j*phaseShifterRad)

overSampFact = 1500
Fs = overSampFact*adcSamplingRate #50e9 # GHz
Ts = 1/Fs

DACFreqBin = 512 # ADC sampling rate scale
dacDC = 10 # DC for the DAC tone

num_samples = np.int32(chirp_time//Ts) #2048
time_s = np.arange(num_samples)*Ts
analogSignalFreqGrid = np.arange(-num_samples//2, num_samples//2,1)*Fs/num_samples
ADCSampledSignalFreqGrid = np.arange(-numChirpSamples//2, numChirpSamples//2,1)*adcSamplingRate/numChirpSamples
fre_vs_time = slope*time_s + Fstart_hz
#chirp_phase = 2*np.pi*(0.5*slope*time_s**2 + Fstart_hz*time_s) + initial_phase_rad;
chirp_phase = 2*np.pi*np.cumsum(fre_vs_time)*Ts + initial_phase_rad
localOscillator = np.exp(1j*chirp_phase)
localOscillatorRxChain = localOscillator.copy()
localOscillatorwithPhaseSweep = localOscillator[None,:]*phaseShifterPhasor[:,None]
# localOscillator_InstFreq = (np.diff(np.unwrap(np.angle(localOscillator)))/(2*np.pi))/Ts;
# localOscillator_fft = np.fft.fft(localOscillatorwithPhaseSweep,axis=1)/num_samples

DACSignalFreq = DACFreqBin*(adcSamplingRate/numChirpSamples) #Hz
DACSignal = 1*(np.exp(1j*2*np.pi*DACSignalFreq*time_s) + np.exp(-1j*2*np.pi*DACSignalFreq*time_s))/2 + dacDC

transmittedSignal = localOscillatorwithPhaseSweep*DACSignal[None,:]

numTargets = 3
targetDistances = np.array([0,2,10]) # in m
# targetDistances = np.array([0,2,0.15]) # in m
rangeBins = (targetDistances//rangeRes)
targetDelays = (2*targetDistances)/lightSpeed
delaySamples = np.round(targetDelays/Ts).astype(np.int32)

receivedSignalVec = np.zeros((numTargets,numPhaseCodes,num_samples),dtype=np.complex128)
for sourceNum in np.arange(numTargets):
    receivedSignalVec[sourceNum,:,delaySamples[sourceNum]::] = transmittedSignal[:,0:num_samples-delaySamples[sourceNum]]

receivedSignal = np.sum(receivedSignalVec,axis=0)

del receivedSignalVec, transmittedSignal

# basebandSignal = receivedSignal * np.conj(localOscillatorRxChain)
basebandSignal = localOscillatorRxChain[None,:] * np.conj(receivedSignal)
# basebandSignalFFT = np.fft.fft(basebandSignal*np.hanning(num_samples)[None,:],axis=1)/num_samples
del localOscillatorRxChain,receivedSignal
downSampledSignal = basebandSignal[:,0::overSampFact]
adcSignal = downSampledSignal[:,0:numChirpSamples]
del basebandSignal
windowFunction = np.hanning(numChirpSamples)
windowedADCSignal = adcSignal*windowFunction[None,:]

rangeFFTSignal = np.fft.fft(windowedADCSignal,axis=1)/numChirpSamples
rangeSpectrum = 20*np.log10(np.abs(rangeFFTSignal))
rangeSpectrumFFTshift = np.fft.fftshift(rangeSpectrum,axes=(1,))

crBin = np.int32(rangeBins[1])
binsToSample = np.array([crBin,DACFreqBin])

phasorSignal = rangeFFTSignal[:,binsToSample]
phaseRad = np.unwrap(np.angle(phasorSignal),axis=0)
phaseDeg = phaseRad*180/np.pi
phaseDegNorm = phaseDeg - phaseDeg[0,:]

# plt.figure(1,figsize=(20,9))
# plt.subplot(1,2,1)
# plt.title('Local oscillator signal: Freq vs Time')
# plt.plot(time_s/(1e-6),fre_vs_time/1e9)
# plt.xlabel('Time (us)')
# plt.ylabel('Freq (GHz)')
# plt.grid(True)
# plt.subplot(1,2,2)
# plt.title('Local oscillator signal: Magnitude spectrum (dB)')
# plt.plot(freq_grid/1e9, 20*np.log10(np.fft.fftshift(np.abs(localOscillator_fft))))
# plt.xlabel('Freq(GHz)')
# plt.grid(True)


# plt.figure(2,figsize=(20,10))
# plt.subplot(1,2,1)
# plt.title('Range Spectrum before ADC sampling')
# plt.plot(20*np.log10(np.abs(basebandSignalFFT[:,0:1000].T)))
# plt.vlines(rangeBins,ymin=20,ymax=160)
# plt.vlines(rangeBins+DACFreqBin,ymin=20,ymax=160)
# plt.grid(True)
# plt.subplot(1,2,2)
# plt.title('Range Spectrum post ADC sampling')
# plt.plot(rangeSpectrum[0:1000])
# plt.vlines(rangeBins,ymin=-160,ymax=20)
# plt.vlines(rangeBins+DACFreqBin,ymin=-160,ymax=20)
# plt.grid(True)

targetFrequencies = rangeBins*adcSamplingRate/numChirpSamples
dacGenFreqPos = DACSignalFreq + targetFrequencies[1::]
dacGenFreqNeg = -DACSignalFreq + targetFrequencies[1::]
plt.figure(3,figsize=(20,10),dpi=200)
plt.title('Range Spectrum post ADC sampling. Fs = ' + str(adcSamplingRate/1e6) + ' MHz')
plt.plot(ADCSampledSignalFreqGrid/1e6,rangeSpectrumFFTshift[0,:])
plt.xlabel('Freq(MHz)')
plt.vlines(DACSignalFreq/1e6,ymin=-160,ymax=20,label='Positive DAC frequency',color='orange')
plt.vlines(-DACSignalFreq/1e6,ymin=-160,ymax=20,label='Negative DAC frequency',color='r')
plt.vlines(targetFrequencies/1e6,ymin=-160,ymax=20,linestyle='dashed',label='Target frequencies')
plt.vlines(dacGenFreqPos/1e6,ymin=-160,ymax=20,linestyle='dashdot',label='DAC + target frequencies',color='orange')
plt.vlines(dacGenFreqNeg/1e6,ymin=-160,ymax=20,linestyle='dashdot',label='- (DAC  - target frequencies)',color='r')
plt.grid(True)
plt.legend()


plt.figure(4,figsize=(20,10),dpi=200)
plt.title('Received phase vs phase code sweep in FMCW + DAC mode')
plt.plot(phaseDegNorm[:,0])
plt.plot(phaseDegNorm[:,1],'o')
plt.grid(True)
plt.xlabel('Chirp number')
plt.ylabel('Phase (Deg)')
plt.legend(['CR tone phase', 'DAC tone phase'])
